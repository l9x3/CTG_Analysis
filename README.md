# CTG_Analysis

## Setup

```bash
pip install -r requirements.txt
```

## Run the analysis

```bash
python Analysis.py --input CTG_Dataset.csv --output-dir output
```

Outputs (summary tables and plots) are written to the `output/` directory.

## Federated evaluation outputs

```bash
python federated_analysis.py \
  --models FedAvg,FedProx,FedNova \
  --rounds 50 \
  --privacy-budgets 0.5,1.0,2.0 \
  --noniid-levels 0.1,0.3,0.5 \
  --clients 10 \
  --output-dir output
```

Federated outputs are written to `output/federated/`:

- `convergence_summary.csv` (table for reports)
  - Columns: `model`, `privacy_budget`, `noniid_level`, `rounds`, `final_accuracy`,
    `best_accuracy`, `best_round`, `mean_accuracy`, `final_loss`, `best_loss`, `mean_loss`
- `convergence_curve.csv` (convergence curves)
  - Columns: `model`, `privacy_budget`, `noniid_level`, `round`, `accuracy`, `loss`
- `privacy_utility.csv` (privacy–utility plot)
  - Columns: `model`, `privacy_budget`, `noniid_level`, `final_round`,
    `final_accuracy`, `final_loss`
- `client_distribution.csv` (client-level boxplots)
  - Columns: `model`, `privacy_budget`, `noniid_level`, `client_id`, `accuracy`, `loss`
- `noniid_sensitivity.csv` (non-IID sensitivity curve)
  - Columns: `model`, `noniid_level`, `privacy_budget`, `final_round`,
    `final_accuracy`, `final_loss`

Plots are saved to `output/federated/plots/` as `convergence_curve.png`,
`privacy_utility.png`, `client_distribution.png`, and `noniid_sensitivity.png`.
