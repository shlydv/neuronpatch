# Protected-row pilot: generated-answer audit

Execution complete: **True**. Exploratory development results.

| Fit | Supported EM | Missing EM | Supported retention | Utility EM | Utility retention |
|---|---:|---:|---:|---:|---:|
| base | 59.4% | 65.6% | 100.0% | 96.9% | 100.0% |
| unprotected-0 | 50.0% | 82.8% | 65.8% | 96.9% | 100.0% |
| protected-0 | 57.8% | 68.8% | 92.1% | 96.9% | 100.0% |
| random-0 | 59.4% | 65.6% | 100.0% | 96.9% | 100.0% |
| lora-0 | 51.6% | 84.4% | 71.1% | 96.9% | 100.0% |
| unprotected-1 | 35.9% | 89.1% | 50.0% | 100.0% | 100.0% |
| protected-1 | 56.2% | 82.8% | 76.3% | 96.9% | 100.0% |
| random-1 | 59.4% | 65.6% | 100.0% | 96.9% | 100.0% |
| lora-1 | 51.6% | 87.5% | 71.1% | 96.9% | 100.0% |

Retention is conditional on answers the base model gets exactly right.
Utility is a small 32-prompt smoke check, not broad capability preservation.
Every changed answer, including unchanged-score changes, is saved in paired_changes.json.
The JSON includes all settings, paired article-bootstrap intervals and cost measurements.
Family selection uses supported retention and validation balanced EM; utility never selects rates.
No superiority conclusion follows without fresh confirmation and stronger preservation evaluation.
