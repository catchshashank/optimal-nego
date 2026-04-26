# Results Summary
## From Optimal Stopping to Optimal Talking

---

## Replication: Manzoor et al. (2025) on NL Negotiations Corpus

| Dimension | Manzoor et al. (2025) | NL Replication (Ours) |
|---|---|---|
| Domain | Outbound telecom sales (Spain) | Buyer-seller house negotiations |
| N conversations | 11,627 calls | 178 NL conversations |
| Outcome rate | 5.5% success | 59% sale (105/178) |
| Avg. duration | 195s (failed: 169s; won: 630s) | 389s (range 46–946s) |
| LLM backbone | GPT-4.1 (fine-tuned, SFT) | GPT-4o (zero-shot logprobs) |
| Decision windows | T=2: {60s, 90s} T=3: {30,60,90} | 90s / 180s (recalibrated) |
| Policy method | Imitation learning | Threshold BI on logprob scores |
| Best config AUC | 0.94 at t=60s | 0.753 at t=90s |
| Sales retained | 130/132 (T=2) | 27/27 (RF1×DP2) |
| Time saved | 36% (T=2) | 1.8% (RF1×DP2) |

**Key replication insight:** Original windows (45s/60s) fail in our corpus (AUC 0.52) because GPT-4o zero-shot requires 90s before signal emerges. The original's 94% AUC at t=60s is a function of fine-tuning + domain alignment, not the framework itself.

---

## Stage 1 — SST Emotional Space

- **23 of 27** emotion dimensions retained (SH-CCA r > 0.05)
- Dropped: admiration, relief, remorse, surprise
- Sparsity improvement: 94.9% → 83.5% (binary + rolling window)
- Active emotion dims per turn: 1.17 → 7.59

**Buyer vs. Seller emotional profiles:**

| Role | Dominant emotions |
|---|---|
| Buyer | Curiosity, confusion, desire, disappointment |
| Seller | Grief, approval, optimism, pride |

---

## Stage 2 — Coupled State-Space Model

**Model selection (BIC):** k=4 latent dimensions

| k | BIC |
|---|---|
| 2 | −499,003 |
| 3 | −510,229 |
| **4** | **−518,944** ← best |

**Key parameters:**
- Spectral radius A: 0.959 (stable, persistent dynamics)
- Transition noise Q mean: 0.190
- Emission noise R mean: 0.014

**Outcome prediction (emotion only, no price):**

| Metric | Value |
|---|---|
| Mean 5-fold CV AUC | 0.670 |
| Overall AUC | 0.657 |
| Sale mean pred. prob. | 0.579 |
| No-sale mean pred. prob. | 0.456 |
| Gap | 0.123 |

**Trajectory finding:** z_1 divergence between sale and no-sale is largest at mid-call (−0.38 vs −0.24), not at resolution. Emotional signature of deal closure is set before Phase 3.

---

## Stage 3 — Bargaining Act Annotation

**Layer 1 coverage (substantive turns, n=5,802):**

| Act | N turns | % |
|---|---|---|
| Comparison | 1,025 | 17.7% |
| New Offer | 819 | 14.1% |
| Repeat Offer | 659 | 11.4% |
| Push | 615 | 10.6% |
| End | 450 | 7.8% |
| Allowance | 162 | 2.8% |
| No act | 3,450 | 59.5% |

**Layer 2 — Push sub-types:**

| Sub-type | N | % of Push |
|---|---|---|
| push_neutral | 523 | 85.0% |
| push_constraint | 75 | 12.2% |
| push_disparagement | 17 | 2.8% |

**Layer 2 — Comparison sub-types:**

| Sub-type | N | % of Comparison |
|---|---|---|
| comparison_price | 547 | 53.4% |
| comparison_quality | 295 | 28.8% |
| comparison_mixed | 183 | 17.9% |

---

## Task 3B — Tactic → Latent State Shift

Mean shift in dominant z-dimension per act (z_t → z_{t+1}):

| Act | N | Dominant dim | Mean shift | Direction |
|---|---|---|---|---|
| push_disparagement | 17 | z_1 | −0.12564 | ↓ toward sale |
| comparison_mixed | 183 | z_3 | +0.08326 | ↑ away from sale |
| end | 397 | z_1 | −0.06675 | ↓ toward sale |
| push_constraint | 74 | z_4 | −0.06146 | ↓ |
| comparison_price | 541 | z_1 | −0.05180 | ↓ toward sale |
| new_offer | 805 | z_1 | −0.04856 | ↓ toward sale |
| comparison | 1,019 | z_1 | −0.04653 | ↓ toward sale |
| comparison_quality | 295 | z_2 | −0.03316 | ↓ |
| push | 608 | z_4 | −0.02431 | ↓ |
| allowance | 153 | z_2 | +0.02254 | ↑ |
| repeat_offer | 609 | z_2 | +0.02164 | ↑ |

**Note on comparison_mixed z_3 shift:** z_3 is the dimension with the largest sale/no-sale gap at mid-call. comparison_mixed is the only act that moves z_3 positively (+0.083) — mixed evidence turns move the dyad upward in the emotionally convergent dimension. This is the core Move 1 finding.

---

## Task 3C — Horse Race Regression

**4-model nested AUC comparison:**

| Model | N features | Mean CV AUC | Std | Overall AUC |
|---|---|---|---|---|
| M1: Price only | 9 | 0.5390 | 0.0654 | 0.5391 |
| M2: SSM (emotion) only | 20 | 0.6269 | 0.1117 | 0.6187 |
| M3: Price + SSM | 29 | 0.5848 | 0.0782 | 0.5847 |
| M4: Price + SSM + Tactics | 47 | 0.6143 | 0.0655 | 0.6038 |

**Marginal contributions:**

| Test | Delta AUC | Interpretation |
|---|---|---|
| SSM adds to price | +0.046 | Emotional channel independent of price |
| Tactics add to price+SSM | +0.030 | Tactic channel independent of emotion |
| Full linguistic channel (M4−M1) | +0.075 | Language adds substantially beyond price |
| Emotion alone vs price (M2−M1) | +0.088 | Emotion is the stronger predictor |

**Main finding:** Language predicts outcome better than price dynamics in tight-ZOPA negotiations. The independent linguistic channel (M4 vs M1) is +0.075 AUC. This constitutes empirical support for the existence of an "optimal talking" layer beyond optimal stopping.

---

## Boundary Conditions

1. **ZOPA width**: tight $10k surplus amplifies language advantage; wide ZOPA would reduce it
2. **Single-issue**: multi-issue B2B negotiations require tactic space extension
3. **Symmetric information**: both parties have same comparable data; asymmetry would change comparison effectiveness
4. **Lab setting**: incentives (~$23/hr) are lower than real-world negotiations
5. **Individualistic culture**: findings may not generalize to collectivistic negotiation contexts
6. **One-shot**: repeat interactions introduce relationship and reputation dynamics
