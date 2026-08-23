# Recovery Intelligence Engine

## Problem

A failed recurring payment does not necessarily mean lost revenue. Likewise,
a payment that succeeds after a recovery intervention does not necessarily
mean that the intervention caused the recovery.

## Goal

Build an AI-assisted decision layer that estimates when recovery intervention
is likely to create additional revenue, selects the minimum justified action,
and evaluates the policy against conventional recovery strategies.

## Scope

Failed recurring/subscription payments only.

## Important limitation

The project uses a controlled synthetic environment with hidden counterfactual
ground truth. Results represent simulated policy comparisons and are not
production causal estimates.

## Repository layout

```
recovery-intelligence-engine/
├── simulator/       Synthetic payment environment generator
│   ├── config.py              Constants, enums, probability tables
│   ├── customer_generator.py  Customer profiles (observable + hidden)
│   ├── subscription_generator.py  Subscription records
│   ├── payment_generator.py   Failed payment records
│   ├── ground_truth.py        Hidden counterfactual probabilities
│   └── generate.py            Main pipeline orchestrator
├── data/raw/        Generated datasets
├── backend/         Decision API (future)
├── models/          Uplift/CATE estimator (future)
├── evaluation/      Baseline policies + evaluation framework (future)
├── frontend/        Dashboard (future)
└── docs/            Architecture notes, assumptions, limitations
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install pandas numpy faker
python simulator/generate.py
```

## License

Internal project — not for redistribution.
