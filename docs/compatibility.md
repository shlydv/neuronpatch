# Runtime compatibility checks

The package declares `torch>=2.4`. A fresh Torch2.4.1 environment exposed an
attachment/export bug: `nn.Module.set_submodule` is unavailable there. Targets
are now replaced through their validated parent module with ordinary module
assignment. Patch formats and the public API are unchanged.

Regression coverage removes the newer API even on modern PyTorch and checks
column, row and grouped-row patches through nested ModuleList/ModuleDict paths,
including save/load and merged-model agreement.

| Local check | Result |
|---|---|
| Clean Python3.11.16, Torch2.4.1, NumPy1.26.4, editable `.[dev]`, no Transformers/PEFT | 152 passed, 25 optional skips |
| Installed modern runtime with Transformers/PEFT | 181 passed, 2 skips |
| Torch2.4.1 with pinned Transformers4.40/PEFT0.11; patch/occurrence/adapter subset | 50 passed |
| Minimum-core toy example | Held-out MSE .00508946 to 2.59987e-12; save/load/merge pass |

These checks run locally on macOS CPU. The new `core-minimum` CI job uses Linux,
Python3.11, Torch2.4.1 and NumPy1.26.4, installs only `.[dev]`, then runs the tests
and toy example. Remote success is determined by its actual Actions result.
The existing modern-runtime integration jobs remain in place.

Pure prompt/scoring and data-preparation helpers also import without Transformers;
their model dependencies load only when the model workflow runs. Tests needing
actual model classes explicitly skip when the optional packages are missing.
The minimal install does not silently download models.

The running Kaggle comparison uses its already accepted frozen source; this
compatibility repair does not change that experiment or its training settings.
