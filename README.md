# 🏏 Rajasthan Royals — IPL 2026 Data-Driven Auction Strategy
### SupeRR Selector Hackathon Submission | Mann Sutariya

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20GradientBoosting-orange.svg)]()
[![Optimization](https://img.shields.io/badge/Optimization-Integer%20Programming-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Hackathon%20Submission-yellow.svg)]()

---

## 📌 Overview

This repository contains my end-to-end data science submission for the **SupeRR Selector Hackathon** organized by Rajasthan Royals (RR), spanning two rounds:

| Round | Problem | Approach |
|-------|---------|----------|
| **Round 1** | Build an optimal IPL 2026 auction strategy for RR under budget, overseas, and slot constraints | Multi-factor impact scoring + Binary Integer Programming |
| **Round 2** | Identify uncapped Indian players most likely to earn an India cap, and predict when | Supervised ML ensemble (Random Forest + Gradient Boosting) |

> ⚠️ **Data Notice:** The raw ball-by-ball and match data was provided privately to hackathon participants and is **not included** in this repository. Only code, derived outputs, feature tables, and model results are published.

---

## 🗂️ Repository Structure

```
RR-IPL-Auction-Strategy/
├── README.md
├── .gitignore
│
├── Round-1/
│   ├── code/
│   │   ├── data_loader.py         # Loads and preprocesses player data
│   │   ├── main.py                # Entry point: runs full auction strategy
│   │   ├── metrics.py             # Batting & bowling impact score calculations
│   │   ├── optimizer.py           # Binary integer programming optimizer
│   │   ├── risk.py                # Risk adjustment framework
│   │   ├── scarcity.py            # Role scarcity weighting
│   │   ├── visualizer.py          # Charts: value scatter, purse Sankey, heatmaps
│   │   └── requirements.txt       # Python dependencies
│   ├── data/                      # ← gitignored (private hackathon data)
│   └── Round1_RR_IPL_2026_Auction_Strategy_Final.pdf
│
└── Round-2/
    ├── code/
    │   ├── cleaning_data.py                    # Raw data cleaning and standardization
    │   ├── column.py                           # Column name utilities
    │   ├── final.py                            # Base feature builder (player master)
    │   ├── final2.py                           # Extended player master creation
    │   ├── smart_loading.py                    # Efficient ZIP-based CSV ingestion
    │   ├── rr2_phase_features.py               # Phase, venue and progression features
    │   ├── build_selection_score.py            # Heuristic SelectionScore baseline
    │   ├── model_selection.py                  # Baseline Random Forest classifier
    │   ├── model_selection_calibrated.py       # Calibrated RF + age decay
    │   ├── model_selection_gb.py               # Gradient Boosting + calibration
    │   ├── rr2_final_assemble.py               # Ensemble (80% RF + 20% GB) → Top-40
    │   ├── rr_predicted_cap_year.py            # Cap year estimation via trajectory
    │   ├── parquet_to_csv.py                   # Convert parquet outputs to CSV
    │   ├── player_into_latex.py                # Export Top-40 table to LaTeX
    │   ├── top20_batting_impact_score.py       # Top-20 batting impact analysis
    │   ├── rr_age_vs_prob_plot.py              # Age vs cap probability plot
    │   ├── rr_prob_distribution_plot.py        # Global probability distribution plot
    │   ├── rr_prob_distribution_tail_zoom.py   # Right-tail zoom plot
    │   ├── rr_role_competition_plot.py         # Role composition comparison plot
    │   ├── trajectory_graph.py                 # Rocket trajectory visualization
    │   │
    │   ├── rr2_prob_distribution_global.png    # Fig 1: Global cap probability dist.
    │   ├── rr2_prob_distribution_tail_zoom.png # Fig 2: Right-tail zoom
    │   ├── rr2_age_vs_cap_probability.png      # Fig 3: Age vs probability
    │   ├── rr2_age_vs_cap_probability_v2.png   # Fig 3 (v2): Age vs probability
    │   ├── rr2_role_composition_all_vs_top40_pct.png  # Fig 4: Role composition
    │   ├── rr2_rocket_trajectory.png           # Fig 5: Rocket trajectory
    │   └── rr2_rocket_trajectory_v2.png        # Fig 5 (v2): Rocket trajectory
    │
    ├── data/                      # ← gitignored (private hackathon data)
    ├── features_out/              # Derived feature tables
    ├── output/                    # Model outputs, Top-40 lists, CSVs
    └── Round2_RR_Hackathon_Final.pdf
```

---

## 🏆 Round 1 — IPL 2026 Auction Strategy

### Problem
Design a data-driven auction strategy for RR under the following hard constraints:
- **Purse:** ₹16.05 Cr
- **Max new signings:** 9 players
- **Max overseas additions:** 1

### Methodology

#### Optimization Objective
A Binary Integer Program that maximizes total squad impact:

```
maximize  Σ Sᵢ · xᵢ
subject to:
  Σ Bᵢ · xᵢ  ≤ 16.05 Cr      (purse constraint)
  Σ xᵢ (overseas) ≤ 1          (overseas constraint)
  Σ xᵢ ≤ 9                     (slot constraint)
  All critical roles covered    (role constraint)
```

#### Scoring Framework
Each player receives a **Final Impact Score** built from five layers:

1. **Batting Impact** — Strike rate, average, boundary %, dot ball %, recent form, clutch performance (knockout/chase scenarios)
2. **Bowling Impact** — Economy, bowling SR, dot ball %, death skill, form, pressure performance
3. **Venue Fit Score** — Tailored to Sawai Mansingh Stadium (spin-friendly middle overs, pace-friendly new ball, dew factor)
4. **Role Scarcity Weighting** — Indian death bowlers (1.40×), wrist spinners (1.35×), left-arm pacers (1.30×)
5. **Risk Adjustment** — Injury risk, availability risk, volatility, and age decline

```
Final Score Sᵢ = ScarcityImpact × (1 − riskᵢ) × VenueFitᵢ
```

#### Final Player Selection

| # | Player | Role | Max Bid | Score |
|---|--------|------|---------|-------|
| 1 | Cameron Green | Pace AR (Overseas) | ₹5.5 Cr | 0.87 |
| 2 | Ravi Bishnoi | Leg-spinner | ₹3.5 Cr | 0.91 |
| 3 | Mohit Sharma | Death Pacer | ₹1.2 Cr | 0.82 |
| 4 | Akash Madhwal | Yorker Specialist | ₹1.0 Cr | 0.84 |
| 5 | Rahul Tripathi | Aggressive Batter | ₹1.5 Cr | 0.79 |
| 6 | Abhinav Manohar | Middle-order Finisher | ₹0.8 Cr | 0.74 |
| 7 | Srikar Bharat | WK-Batter | ₹0.75 Cr | 0.72 |
| 8 | Mahipal Lomror | Batting AR | ₹0.8 Cr | 0.76 |
| 9 | R. Hangargekar | Young Pace AR | ₹0.95 Cr | 0.71 |

**Total: ₹16.00 Cr | Purse Utilization: 99.7%**

#### Projected Impact
| Metric | Improvement |
|--------|-------------|
| Death bowling reliability | +35% |
| Middle-overs wicket-taking | +33% |
| Batting depth (Indian options) | +3 quality additions |
| Yorker % in death overs | +41.2% |

#### Model Validation
Back-tested against IPL 2024 auction outcomes — **76% overall accuracy**, with high-scorers (>0.80) validated at 78% accuracy.

---

## 🔬 Round 2 — Uncapped-to-Capped Prediction Model

### Problem
From a pool of 5,484 players across domestic and T20 competitions, identify the **Top-40 uncapped players** most likely to earn an India cap, and estimate **when** they are likely to be capped.

### ML Pipeline

```
cleaning_data.py / smart_loading.py
        ↓  (clean & ingest 6 CSVs from ZIP)
final.py / final2.py
        ↓  (player master + career / 12m / 24m features)
rr2_phase_features.py
        ↓  (phase-wise, venue, progression features)
build_selection_score.py
        ↓  (heuristic SelectionScore — interpretable baseline)
model_selection.py
        ↓  (baseline Random Forest classifier)
model_selection_calibrated.py
        ↓  (calibrated RF + age decay)
model_selection_gb.py
        ↓  (feature-pruned Gradient Boosting + calibration)
rr2_final_assemble.py
        ↓  (ensemble: 80% RF + 20% GB → Top-40)
rr_predicted_cap_year.py
        ↓  (cap year estimation via trajectory slopes)
```

### Feature Engineering Highlights

| Feature Group | Details |
|--------------|---------|
| Career metrics | Runs, average, SR, economy, wickets, boundary %, dot % |
| Recency windows | Separate 12-month and 24-month aggregates |
| Phase-wise splits | Powerplay / middle / death batting and bowling stats |
| Pressure performance | Close games (margin ≤ 15), knockout matches, chases |
| Venue adaptability | Player performance vs. venue baseline ratio |
| Progression slopes | Seasonal linear trend in runs and wickets (OLS) |
| Role inference | From batting order position and bowling usage |

### Ensemble Model

```
p_final = 0.80 × p_RF_calibrated_age + 0.20 × p_GB_calibrated_age
```

- **Random Forest** (400 trees, isotonic calibration) — stable, handles class imbalance via `balanced_subsample`
- **Gradient Boosting** (HistGBM, sigmoid calibration) — captures non-linear interactions and recent form
- Both models apply **age-based probability decay** for players above 32

### Cap Year Estimation

```
growth_rate gᵢ  = 0.04 + 0.02 × runs_slope_norm + 0.02 × wk_slope_norm   [clipped 0.02–0.12]
years_to_cap    = ceil((0.80 − p_final) / gᵢ)                             [clipped 0–5]
predicted_cap_year = 2025 + years_to_cap
```

### Top-10 from Final Shortlist

| # | Player | Role | Cap Prob | Cap Year | Age |
|---|--------|------|----------|----------|-----|
| 1 | R Sonu Yadav | Finishing AR | 0.76 | 2027 | 26.0 |
| 2 | Baba Indrajith | Top/Middle | 0.75 | 2027 | 31.4 |
| 3 | Abdul Basith | Finishing AR | 0.73 | 2027 | 27.1 |
| 4 | BR Sharath | Top/Middle | 0.72 | 2027 | 29.2 |
| 5 | Manoj Bhandage | Finishing AR | 0.72 | 2030 | 27.1 |
| 6 | Himmat Singh | Top/Middle | 0.72 | 2026 | 29.1 |
| 7 | Shivam Singh | Top/Middle | 0.72 | 2027 | 30.0 |
| 8 | Karan Sharma | Opener | 0.71 | 2027 | 27.1 |
| 9 | Shubhang Hegde | Finishing AR | 0.71 | 2028 | 24.7 |
| 10 | KL Shrijith | Top/Middle | 0.71 | 2029 | 29.3 |

### Visual Outputs

All diagnostic plots are included in `Round-2/code/`:

| File | Description |
|------|-------------|
| `rr2_prob_distribution_global.png` | Cap probability distribution across all 5,484 players |
| `rr2_prob_distribution_tail_zoom.png` | Right-tail zoom: all players vs Top-40 |
| `rr2_age_vs_cap_probability_v2.png` | Age vs cap probability with Top-40 highlighted |
| `rr2_role_composition_all_vs_top40_pct.png` | Role composition: full pool vs shortlist |
| `rr2_rocket_trajectory_v2.png` | Rocket trajectory: historical capped vs projected Top-40 |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn (RandomForest, HistGradientBoosting, CalibratedClassifierCV) |
| Optimization | Binary Integer Programming (custom formulation) |
| Visualization | matplotlib |
| Document Preparation | LaTeX |
| Data Formats | CSV, Parquet |

---

## ▶️ How to Run

> **Note:** Raw data files are not included. You need access to the hackathon dataset (`compressed_files.zip`) to run the full pipeline.

```bash
# Clone the repo
git clone https://github.com/<your-username>/RR-IPL-Auction-Strategy.git
cd RR-IPL-Auction-Strategy

# Install dependencies (Round 1)
pip install -r Round-1/code/requirements.txt

# Run Round 1
cd Round-1/code
python main.py

# Install dependencies (Round 2)
pip install pandas numpy scikit-learn matplotlib pyarrow

# Run Round 2 pipeline (in order)
cd ../../Round-2/code
python cleaning_data.py
python final2.py
python rr2_phase_features.py
python build_selection_score.py
python model_selection.py
python model_selection_calibrated.py
python model_selection_gb.py
python rr2_final_assemble.py
python rr_predicted_cap_year.py
```

---

## 📄 Documents

- 📘 [Round 1 — Auction Strategy (PDF)](Round-1/Round1_RR_IPL_2026_Auction_Strategy_Final.pdf)
- 📗 [Round 2 — Uncapped Player Prediction (PDF)](Round-2/Round2_RR_Hackathon_Final.pdf)

---

## 👤 Author

**Mann Sutariya**
📧 mannsutaria2605@gmail.com
📱 +91 7984342097
📅 November 2025

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).

The underlying ball-by-ball dataset is proprietary to the SupeRR Selector Hackathon organizers and is not redistributed here.
