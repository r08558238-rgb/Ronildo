"""
DACM v6.0 	6 Minimalist Functional Release

The definitive, production-ready, highly polished version.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import warnings

# =====================================================
# Configuration
# =====================================================
class Config:
    ENV_NAME = "LunarLanderContinuous-v2"
    N_ENVS = 4
    TOTAL_TIMESTEPS = 100_000

    LATENT_DIM = 64
    NUM_LATENTS = 16
    PERCEIVER_BLOCKS = 2

    BOTTLENECK_DIM = 16
    VALENCE_DIM = 3
    INTENSITY_DIM = 1
    LSTM_HIDDEN = 128
    FEATURES_DIM = LSTM_HIDDEN

    CONSCIOUSNESS_THRESHOLD = 0.70
    ALPHA_ALIGN = 0.25

    POLICY_KWARGS = dict(
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    SEED = 42

config = Config()
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")

# =====================================================
# Perceiver Block
# =====================================================
class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, latents: th.Tensor, context: th.Tensor) -> th.Tensor:
        attn_out, _ = self.attn(self.norm1(latents), context, context)
        x = latents + attn_out
        x = x + self.ff(self.norm2(x))
        return x

# =====================================================
# Perceiver IO
# =====================================================
class PerceiverIO(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.LATENT_DIM)
        self.latents = nn.Parameter(th.randn(1, config.NUM_LATENTS, config.LATENT_DIM) * 0.02)
        self.blocks = nn.ModuleList([PerceiverBlock(config.LATENT_DIM) for _ in range(config.PERCEIVER_BLOCKS)])

    def forward(self, x: th.Tensor, feedback: Optional[th.Tensor] = None) -> th.Tensor:
        bsz = x.size(0)
        ctx = self.proj(x).unsqueeze(1)
        lat = self.latents.repeat(bsz, 1, 1)
        if feedback is not None:
            lat = lat + feedback.unsqueeze(1)
        for block in self.blocks:
            lat = block(lat, ctx)
        return lat.mean(dim=1)

# =====================================================
# Bottleneck and Affective Heads
# =====================================================
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(config.LATENT_DIM, 96),
            nn.ReLU(),
            nn.Linear(96, config.BOTTLENECK_DIM),
        )
        self.valence_head = nn.Linear(config.BOTTLENECK_DIM, config.VALENCE_DIM)
        self.intensity_head = nn.Linear(config.BOTTLENECK_DIM, config.INTENSITY_DIM)
        self.align_proj = nn.Linear(config.BOTTLENECK_DIM, config.LATENT_DIM)

    def forward(self, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        ppc_bottleneck = self.core(z)
        valence = th.tanh(self.valence_head(ppc_bottleneck))
        intensity = th.sigmoid(self.intensity_head(ppc_bottleneck))
        conscious = (intensity > config.CONSCIOUSNESS_THRESHOLD).float()
        ppc_aligned = self.align_proj(ppc_bottleneck)
        return ppc_bottleneck, ppc_aligned, valence, intensity, conscious

# =====================================================
# Alignment Loss
# =====================================================
class AlignmentLoss(nn.Module):
    def forward(self, z_a: th.Tensor, ppc_aligned: th.Tensor) -> th.Tensor:
        z_a_norm = F.normalize(z_a, p=2, dim=-1)
        ppc_norm = F.normalize(ppc_aligned, p=2, dim=-1)
        return 1.0 - F.cosine_similarity(z_a_norm, ppc_norm, dim=-1).mean()

# =====================================================
# DACM Feature Extractor
# =====================================================
class DACMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]

        self.perceiver_a = PerceiverIO(obs_dim)
        self.perceiver_b = PerceiverIO(obs_dim)
        self.bottleneck = Bottleneck()
        self.align_loss_fn = AlignmentLoss()

        lstm_input_dim = config.BOTTLENECK_DIM + config.VALENCE_DIM + config.INTENSITY_DIM
        self.lstm = nn.LSTM(lstm_input_dim, config.LSTM_HIDDEN, batch_first=True)
        self.output_head = nn.Linear(config.LSTM_HIDDEN, features_dim)

        self.register_buffer("conscious_count", th.tensor(0.0))
        self.register_buffer("total_steps", th.tensor(0.0))
        self._aux_data: Dict[str, th.Tensor] = {}

        self.to(DEVICE)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs = observations.to(DEVICE)
        bsz = obs.size(0)

        z_a = self.perceiver_a(obs)
        z_b = self.perceiver_b(obs, feedback=z_a.detach())

        ppc_b, ppc_a, valence, intensity, conscious = self.bottleneck(z_b)

        align_loss = self.align_loss_fn(z_a.detach(), ppc_a)

        gated_ppc = ppc_b * conscious
        lstm_input = th.cat([gated_ppc, valence, intensity], dim=-1).unsqueeze(1)

        # Stateless LSTM
        h0 = th.zeros(1, bsz, config.LSTM_HIDDEN, device=DEVICE)
        c0 = th.zeros(1, bsz, config.LSTM_HIDDEN, device=DEVICE)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        features = self.output_head(lstm_out.squeeze(1))

        self.conscious_count += conscious.sum().detach()
        self.total_steps += bsz
        conscious_rate = self.conscious_count / th.clamp(self.total_steps, min=1.0)

        self._aux_data = {
            "alignment_loss": align_loss, # CRITICAL: Loss tensor must NOT be detached here
            "valence_mean": valence.mean().detach(),
            "intensity_mean": intensity.mean().detach(),
            "conscious_rate": conscious_rate.detach(),
        }

        return features

# =====================================================
# DACM Policy
# =====================================================
class DACMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=DACMExtractor,
            features_extractor_kwargs=dict(features_dim=config.FEATURES_DIM),
        )
        self._aux_loss: Optional[th.Tensor] = None
        self._aux_dict: Dict[str, Any] = {}

    @property
    def aux(self):
        return self._aux_dict

    def forward(self, obs, deterministic=False):
        actions, values, log_probs = super().forward(obs, deterministic)

        aux = self.features_extractor._aux_data
        
        align_loss = aux.get("alignment_loss", th.tensor(0.0, device=self.device))
        self._aux_loss = config.ALPHA_ALIGN * align_loss

        self._aux_dict = {
            k: v.cpu().item() if isinstance(v, th.Tensor) else v
            for k, v in aux.items() if k != "alignment_loss"
        }
        self._aux_dict["weighted_align_loss"] = self._aux_loss.detach().cpu().item()

        return actions, values, log_probs

    def _extra_losses(self):
        return [self._aux_loss] if self._aux_loss is not None else []

# =====================================================
# DACM Logging Callback
# =====================================================
class DACMCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % config.N_ENVS == 0:
            aux = self.model.policy.aux
            if aux:
                for k, v in aux.items():
                    self.logger.record(f"dacm/{k}", float(v))
        return True

    def _on_rollout_end(self):
        self.model.policy.features_extractor.conscious_count.zero_()
        self.model.policy.features_extractor.total_steps.zero_()

# =====================================================
# Environment Setup
# =====================================================
def make_env():
    return Monitor(gym.make(config.ENV_NAME))

# =====================================================
# Main Training Loop
# =====================================================
def main():
    th.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    print("="*60)
    print("DACM v6.0 	6 Minimalist Functional Release")
    print(f"Device: {DEVICE}")
    print(f"Alignment Loss is ACTIVE (weight={config.ALPHA_ALIGN})")
    print("="*60)

    warnings.filterwarnings("ignore", category=UserWarning)

    env = DummyVecEnv([make_env for _ in range(config.N_ENVS)])
    callback = DACMCallback(verbose=0)

    model = PPO(
        DACMPolicy,
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log="./dacm_v60_tb/",
        seed=config.SEED,
        **config.POLICY_KWARGS,
    )

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callback,
            log_interval=1,
            tb_log_name="dacm_v60_run",
        )
        model.save("dacm_v60_final")
    except KeyboardInterrupt:
        model.save("dacm_v60_interrupted")
        print("\nTraining interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
