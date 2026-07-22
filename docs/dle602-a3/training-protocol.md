# ReviewPulse v3 neural training protocol

This protocol is the experiment contract for ReviewPulse v3. It applies to the target-agnostic LSTM, ATAE-LSTM and DistilBERT sentence-pair models before the four-model comparison in issue #84.

## Reproducibility controls

Every run seeds Python, NumPy and PyTorch before the split, model and shuffled `DataLoader` are created. CUDA receives the same seed when available, and cuDNN benchmarking is disabled in favour of deterministic kernels. The official test partition is never used for model selection.

The saved run configuration records:

- model and pretrained model identifier where applicable;
- seed and active device;
- requested and completed epochs;
- batch size, learning rate, optimiser and weight decay;
- early-stopping patience;
- review and aspect maximum sequence lengths where applicable.

Default training parameters are:

| Model | Optimiser | Learning rate | Weight decay | Epochs | Batch | Max length | Device |
|---|---|---:|---:|---:|---:|---:|---|
| Target-agnostic LSTM | Adam | 0.001 | 0.0001 | 8 | 64 | 80 | CPU |
| ATAE-LSTM | Adam | 0.001 | 0.0001 | 8 | 64 | 80 review / 12 aspect | CPU |
| DistilBERT | AdamW | 0.00002 | 0.01 | 2 | 8 | 128 | CUDA, then MPS, then CPU |

Final values used by reported experiments come from the saved run configuration, not from this defaults table.

## Development selection and early stopping

Each epoch records mean training loss and development macro-F1. A checkpoint replaces the current best state only when development macro-F1 improves. Training stops after the configured number of consecutive non-improving epochs, and the selected best state is restored before development and official-test metrics are generated.

The default patience is two epochs. `best_epoch`, `best_development_macro_f1`, `selection_metric` and `stopped_early` are saved with the complete epoch history.

## Overfitting and multi-seed decision

The run record flags material within-run overfitting when development macro-F1 drops by at least 0.02 after its best epoch while training loss continues to fall. This diagnostic does not itself estimate between-seed variance.

- If the flag is false and repeated execution shows no instability, the fixed seed is retained to protect the zero-cost compute budget.
- If the flag is true, or later runs show material instability, issue #84 must run and report multiple seeds before drawing comparative conclusions.

The decision and measured post-best score drop are persisted in `overfitting_diagnostic`; they must not be inferred retrospectively from the final test result.

## Persisted evidence

- `target_lstm.pt` and `atae_lstm.pt` embed the run configuration, history and checkpoint-selection metadata alongside the state dictionary and vocabulary.
- `distilbert/training_run.json` stores the same metadata beside the Hugging Face checkpoint.
- Each trainer also writes its complete `<model>_metrics.json`, including selected development metrics, official test metrics and the overfitting diagnostic.

Issue #84 must consume only restored best checkpoints and preserve these records with the generated comparison artifacts.
