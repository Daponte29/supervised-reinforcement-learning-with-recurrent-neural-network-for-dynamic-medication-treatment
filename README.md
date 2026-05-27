# Dynamic Medication Treatment Recommendation with SRL-RNN

This project replicates the "Supervised Reinforcement Learning with Recurrent Neural Network (SRL-RNN) for Dynamic Medication Treatment" method. It recommends per-timestep medication sets for ICU admissions from MIMIC-III by combining Supervised Learning (imitating clinician prescriptions) with Reinforcement Learning (an Actor-Critic policy guided by survival-based rewards). A plain LSTM baseline is included for comparison.

## Project Structure

```
.
├── data/                  raw + processed MIMIC-III matrices (pickles, CSVs)
├── data_preprocessing/    scripts that build the processed matrices
├── configs/               YAML experiment configs (base, experiment_01)
├── src/                   core library
│   ├── config.py          dataclass configs + YAML loader
│   ├── data.py            matrix loading, alignment, MedicalDataset, collate, loaders
│   ├── model.py           ActorNetwork, CriticNetwork, BasicLSTM, soft_update
│   ├── loss.py            loss registry (BCE for the sigmoid policy head)
│   ├── metrics.py         Jaccard / accuracy / AverageMeter
│   ├── train.py           actor-critic + baseline training loops, CLI
│   ├── evaluate.py        test-set evaluation CLI
│   ├── utils.py           seeding, device, checkpoint I/O
│   └── plots.py           training-curve plots
├── tests/                 unit + smoke tests (run on synthetic data, no MIMIC needed)
├── output_models/         saved checkpoints (.pth)
└── archive/               original main.ipynb (reference only)
```

## Setup

```bash
pip install -e ".[dev]"
# or: conda env create -f environment.yml
```

## Usage

```bash
make train         # train the Actor-Critic SRL-RNN  (configs/experiment_01.yaml)
make train-lstm    # train the Basic LSTM baseline
make evaluate      # evaluate a checkpoint on the test set
make test          # run the test suite
make lint          # ruff + mypy
```

Run modules directly for finer control:

```bash
python -m src.train --config configs/experiment_01.yaml --model actor_critic
python -m src.evaluate --config configs/experiment_01.yaml --checkpoint output_models/best_actor.pth
```

## Data

The training pipeline expects the processed per-admission matrices (`treatment_matrices`,
`time_series_matrices_filtered`, `static_matrices_filtered`, `diagnosis_matrices`, and
`filtered_df_admission_LABELS`) in `data/`. Regenerate them with the scripts in
`data_preprocessing/` if needed.

## Models


1. **SRL-RNN (Actor-Critic)** — the proposed model. The Actor (LSTM over labs + embedded
   diagnoses + demographics) proposes a medication set; the Critic estimates Q-values, and
   the Actor is trained with a mix of `-Q` (RL) and BCE imitation (SL) losses, with Polyak
   target updates.
2. **Basic LSTM** — a purely supervised baseline trained with BCE.

Both are evaluated with sample-averaged **Jaccard** (drug-set overlap vs. clinicians) and
subset accuracy.

## Results

| Model | Jaccard Score |
|----|----|
| SRL-RNN |    |
| Basic LSTM |    |


