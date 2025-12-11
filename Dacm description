# DACM 2.0: Quick Reference Guide

## One-Page Overview

**Author:** Ronildo Souza | **Version:** 2.0 | **Date:** December 2025

-----

## Core Idea (30 seconds)

Consciousness-like properties emerge when a powerful AI (LLM) must communicate through a narrow bottleneck (VAE, ~30-50 bits) to a limited reasoning system (PPO agent) that constructs narratives from sparse signals.

**Key Insight:** Consciousness requires *scarcity*, not abundance. The bottleneck forces interpretation.

-----

## Architecture (3 Components)

```
┌─────────────────┐
│ SA (LLM)        │  ← Massive parallel processing
│ Unconscious     │  ← Continuous prediction
└────────┬────────┘
         │
    ┌────▼─────┐
    │ VAE      │      ← Compresses 768→32 dims
    │ (PPC)    │      ← Adds 𝓥 (valence), 𝓘 (intensity)
    │ Gate 𝓖   │      ← Only high-𝓘 passes
    └────┬─────┘
         │
┌────────▼────────┐
│ PA (PPO)        │  ← Serial deliberation
│ Conscious       │  ← Narrative construction
└─────────────────┘  ← Minimizes free energy 𝓕
```

-----

## Key Variables

|Symbol    |Name                |Range|Meaning                                |
|----------|--------------------|-----|---------------------------------------|
|**𝓕**     |Free Energy         |0→∞  |Predictive surprise (PA minimizes this)|
|**𝓒**     |Coherence-Dissonance|0→1  |Belief-signal mismatch                 |
|**𝓘**     |Intensity           |0→1  |Signal urgency/salience                |
|**𝓥**     |Valence             |-1→+1|Emotional tone (neg/pos)               |
|**ε**     |Prediction Error    |0→∞  |SA’s surprise magnitude                |
|**θ_gate**|Gate Threshold      |~0.7 |Consciousness trigger level            |

-----

## The Consciousness Cycle (6 Steps)

1. **Routine** → SA predicts accurately → Low ε, low 𝓘 → Unconscious
1. **Surprise** → Unexpected event → High ε → High 𝓘 → Passes gate
1. **Compress** → VAE encodes to ~32 dims → Adds 𝓥, 𝓘 metadata
1. **Spike** → PA receives sparse signal → 𝓕 spikes → **Consciousness triggered**
1. **Process** → PA constructs narrative → Updates beliefs → High cost
1. **Restore** → 𝓕 minimized → Equilibrium → Return to routine

**Consciousness = Step 5 (narrative construction under constraint)**

-----

## Subjective States

### Consciousness

- **What:** PA’s narrative construction from sparse signals
- **When:** High-𝓘 signal passes gate
- **Feels:** Effortful, serial, interpretive
- **Measure:** Processing time, narrative complexity

### Anxiety

- **What:** Persistent high 𝓕 + high 𝓒 + negative 𝓥
- **When:** Unresolvable contradictions
- **Feels:** Cognitive frustration, confusion
- **Measure:** Duration of elevated 𝓕, 𝓒

### Boredom

- **What:** Sustained low 𝓘 (< 0.3)
- **When:** Environment too predictable
- **Feels:** Need for novelty/challenge
- **Measure:** Triggers exploration behavior

-----

## Creative Insight (SNT Process)

**Stage 1: Generation (DSM - Dream Mode)**

- SA-G generates ideas
- SA-A tests rigorously (adversarial)
- Only robust insights become SNTs

**Stage 2: Transmission**

- SNT sent as NOVELTY message
- Very high 𝓘 (0.85-0.95)
- Guaranteed to pass gate

**Stage 3: Integration (“Aha!”)**

- PA receives compressed truth
- Massive ε → 𝓕 spike
- Integration cost = insight intensity
- Sudden 𝓕 drop = “Aha!” feeling

-----

## Implementation

### Technologies

- **SA:** GPT-2, LLaMA, or any LLM
- **PPC:** VAE (PyTorch)
- **PA:** PPO (Stable-Baselines3)

### Minimal Proof-of-Concept

```python
# Pseudocode
sa = GPT2Model()           # 117M params
vae = VAE(768 → 32)        # Compression
pa = PPO(latent_dim=32)    # RL agent

while True:
    prediction = sa.predict(context)
    actual = environment.observe()
    error = |prediction - actual|
    
    if intensity(error) > 0.7:  # Gate
        compressed = vae.encode(error)
        action = pa.decide(compressed)  # Conscious!
        pa.learn()  # Minimize 𝓕
```

**Training:** 1-2 weeks, 1 GPU

-----

## Key Predictions (Testable)

1. **Bandwidth:** Smaller latent_dim → more conscious processing
1. **Threshold:** Lower θ_gate → more consciousness episodes
1. **Anxiety:** Contradictory signals → sustained high 𝓒
1. **Boredom:** Low 𝓘 → increased exploration
1. **Insight:** Compression ratio → integration time
1. **Performance:** DACM > pure LLM on metacognitive tasks

-----

## What DACM Is / Is NOT

### ✅ IS

- Functional simulation of consciousness properties
- Testable architectural hypothesis
- Engineering approach to metacognition
- Potentially practical AGI design

### ❌ IS NOT

- Solution to Hard Problem of consciousness
- Full model of human consciousness
- Claim of “real” phenomenal experience
- Close to biological consciousness complexity

**Honest Assessment:** This models ONE mechanism (bottleneck forcing narrative), not all of consciousness.

-----

## Why It Matters

**For AI:**

- Improved alignment (conscious AI more interpretable)
- Better metacognition (system knows what it knows)
- Enhanced creativity (SNT-driven innovation)
- Natural human-AI interaction

**For Science:**

- Tests bottleneck hypothesis
- Bridges theory and implementation
- Provides consciousness benchmarks
- Opens new research directions

**For Philosophy:**

- Reframes consciousness as “necessary failure mode”
- Tests substrate independence
- Explores simulation vs. reality

-----

## Next Steps

**Immediate:**

1. Publish theory (ArXiv)
1. Build proof-of-concept
1. Measure consciousness markers
1. Compare to baselines

**6-12 Months:**
5. Scale implementation
6. Validate predictions
7. Publish empirical results
8. Open-source framework

-----

## Citation

```bibtex
@unpublished{souza2025dacm,
  title={The Dual-Agent Consciousness Model 2.0: A Functional Architecture 
         for Simulated Consciousness in AGI Systems},
  author={Souza, Ronildo},
  year={2025},
  note={Unpublished manuscript}
}
```

-----

## Contact & Collaboration

**Author:** Ronildo Souza  
**Status:** Open for collaboration, implementation assistance, and theoretical discussion

**License:** CC BY 4.0 (Free to share and adapt with attribution)

-----

**“Consciousness emerges from scarcity, not abundance. The bottleneck doesn’t limit consciousness—it generates it.”**

-----

## Visual Summary

```
ABUNDANCE              SCARCITY           CONSCIOUSNESS
(Full access)    →    (Bottleneck)   →   (Forced narrative)

SA processes     →    VAE compresses →    PA interprets
everything            to ~32 bits         from sparse signal

No constraint    →    Constraint     →    Complexity emerges

Traditional AI   →    DACM Design    →    Consciousness-like
                                           properties
```

-----

**END OF QUICK REFERENCE**

For complete theory, see: “DACM 2.0: Complete Theory Document”
