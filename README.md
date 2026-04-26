# From Optimal Stopping to Optimal Talking
## Conversational Dynamics in Sales Negotiations Using LLMs

**Shashank Dubey · HEC Paris · PhD Day 2026**

---

## Overview

This repository implements a three-stage computational pipeline for studying how conversational tactics drive emotional dynamics and outcomes in bilateral price negotiations. It replicates and extends Manzoor, Ascarza & Netzer (2025) on the Stanford NL Negotiations Corpus (Heddaya et al. 2024), then applies the framework to proprietary car dealership negotiation transcripts.

**Core research question:** Beyond *when* to quit a sales call, *what to say* — and in what sequence — drives dealer concessions?

---

## Repository Structure

```
optimal-talking/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── nego-data-final.csv          # 178 NL negotiation transcripts (Heddaya et al. 2024)
│
├── stage1/
│   ├── stage1_sst_embedding.py      # Semantic Space Theory emotion projection
│   ├── stage1_turns_embedded.csv    # 5,802 turns × 23 SST emotion dims
│   ├── stage1_sst_dimensions.csv    # Split-half reliability per dimension
│   ├── stage1_backchannels.csv      # 1,763 regulatory turns (preserved for SSM)
│   └── stage1_diagnostics.txt
│
├── stage2/
│   ├── stage2_ssm.py                # Coupled linear Gaussian SSM (EM)
│   ├── stage2_latent_states.csv     # z_t per turn per conversation (k=4)
│   ├── stage2_trajectories.csv      # Dense 46-dim buyer×seller observations
│   ├── stage2_outcome_prediction.csv
│   ├── stage2_ssm_params.npz        # Fitted A, C, Q, R matrices
│   └── stage2_diagnostics.txt
│
├── stage3/
│   ├── stage3_annotation.py         # Bargaining act annotation (Heddaya et al.)
│   ├── stage3_annotated.csv         # 7,565 turns with Layer 1 + Layer 2 labels
│   ├── stage3_diagnostics.txt
│   ├── task3bc.py                   # Tactic×latent overlay + horse race regression
│   ├── task3b_tactic_shifts.csv     # z_t → z_{t+1} shift per act
│   ├── task3c_horse_race.csv        # 4-model AUC comparison
│   ├── task3c_features.csv          # Conversation-level feature matrix
│   └── task3bc_diagnostics.txt
│
└── replication/
    └── replication-deck-v2.pptx     # 5-slide comparison: Manzoor et al. vs NL corpus
```

---

## Pipeline

### Stage 1 — Semantic Space Theory Embedding
`stage1/stage1_sst_embedding.py`

Projects each turn into a 23-dimensional emotional semantic space following Keltner, Brooks & Cowen (2023). Uses binary seed-word detection + rolling window smoothing (window=3 turns) on the GoEmotions 27-category label space. Dimensionality selected via Split-Half CCA (SH-CCA), retaining dimensions with r > 0.05.

**Key outputs:**
- 23 retained emotional dimensions (4 dropped: admiration, relief, remorse, surprise)
- Sparsity reduced from 94.9% (TF-IDF) to 83.5% (binary + rolling)
- Buyer and seller turns tracked separately

### Stage 2 — Coupled State-Space Model
`stage2/stage2_ssm.py`

Fits a linear Gaussian SSM via Expectation-Maximisation on 46-dimensional coupled buyer-seller observation sequences. Latent dimension k=4 selected by BIC. Captures joint dyadic emotional configuration z_t at each turn.

**Key outputs:**
- Spectral radius A = 0.959 (stable, persistent dynamics)
- Outcome prediction AUC = 0.657 (emotion only, no price)
- Sale conversations show steeper z_1 decline at mid-call (convergence signature)

### Stage 3 — Bargaining Act Annotation + Analysis
`stage3/stage3_annotation.py` and `stage3/task3bc.py`

**Layer 1:** Heddaya et al. (2024) six bargaining acts — New Offer, Repeat Offer, Push, Comparison, Allowance, End — annotated via rule-based parser. Multi-label per turn.

**Layer 2:** Sub-typing within Push (Lee & Ames 2017):
- `push_constraint`: References own limitation ("can't", "budget", "afford")
- `push_disparagement`: Attacks counterpart's position ("overpriced", "not worth")
- `push_neutral`: Neither

Sub-typing within Comparison:
- `comparison_price`: Cites external selling prices
- `comparison_quality`: Cites property attributes
- `comparison_mixed`: Both

**Task 3B — Tactic × Latent Shift:**
Computes z_{t+1} − z_t for each act type to identify which conversational moves shift the dyadic emotional state most.

**Task 3C — Horse Race Regression (4 nested models):**

| Model | Features | Mean AUC |
|---|---|---|
| M1 Price only | 9 | 0.539 |
| M2 SSM (emotion) only | 20 | 0.627 |
| M3 Price + SSM | 29 | 0.585 |
| M4 Price + SSM + Tactics | 47 | 0.614 |

**Linguistic channel adds +0.075 AUC over price alone.**

---

## Key Findings

1. **Language predicts outcome better than price** (M2 AUC 0.627 > M1 AUC 0.539) in tight-ZOPA negotiations
2. **Comparison-Mixed is the highest-value act**: only act causing a positive z_3 shift (+0.083), disproportionately present in sale conversations (+3.0pp)
3. **Push-Disparagement is the highest-risk act**: largest emotional shift in corpus (−0.126 on z_1), negatively associated with sale
4. **Critical window is mid-call, not resolution**: z_1 divergence between sale/no-sale is largest at 33–66% of call
5. **Emotional trajectories are coupled and path-dependent**: spectral radius 0.959 means early moves set the dyadic configuration

---

## Theoretical Grounding

| Component | Source |
|---|---|
| Optimal stopping framework | Manzoor, Ascarza & Netzer (2025) |
| Corpus + bargaining act taxonomy | Heddaya, Dworkin, Tan, Voigt & Zentefis (2024) |
| Semantic Space Theory | Keltner, Brooks & Cowen (2023) |
| Contentiousness vs problem-solving | De Dreu, Weingart & Kwon (2000) |
| Constraint vs disparagement rationales | Lee & Ames (2017) |
| Knightian uncertainty / multiple priors | Gilboa, Postlewaite & Schmeidler (2008) |

---

## Requirements

```
python >= 3.10
numpy
pandas
scipy
scikit-learn
```

No GPU required. No proprietary API keys. All models are pure numpy/sklearn.

---

## Running the Pipeline

```bash
# Place nego-data-final.csv in data/
# Run stages in order:

python stage1/stage1_sst_embedding.py
python stage2/stage2_ssm.py
python stage3/stage3_annotation.py
python stage3/task3bc.py
```

Each script reads from outputs of the previous stage. All outputs are written to their respective stage directories.

---

## Citation

```bibtex
@misc{dubey2026optimtalking,
  title  = {From Optimal Stopping to Optimal Talking:
             Conversational Dynamics in Sales Negotiations},
  author = {Dubey, Shashank},
  year   = {2026},
  note   = {HEC Paris Working Paper}
}
```

Data from Heddaya et al. (2024), available at: https://mheddaya.com/research/bargaining
