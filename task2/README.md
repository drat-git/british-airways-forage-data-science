# Task 2: Predicting customer booking completion

Start with the [executed notebook](Booking%20Analysis%20-%20Recommended%20Changes.ipynb) or the [one-slide PowerPoint summary](submission/Booking%20Behaviour%20Summary.pptx). Setup and inference examples are in the [project README](../README.md).

| File or directory | Purpose |
| --- | --- |
| `data/customer_booking.csv` | Original Forage dataset |
| `booking_model.py` | Raw-input validation and selected-model loader |
| `model/selected/` | Selected CatBoost, cutoff, predictions and integrity manifest |
| `model/forest_benchmark/` | Preserved historical forest and evaluation assets |
| `results/verified_results.json` | Metrics, model comparisons, experiments and permutation importance |
| `resources/` | Original starter notebook and assignment notes |
| `requirements.txt` | Pinned analysis dependencies |
| `requirements-lock.txt` | Complete recorded macOS/Python environment |

The notebook refits development comparisons on Run All, then exports the selected CatBoost model. Original CatBoost settings remain selected: neither of the two additional regularisation/learning-rate experiments met the pre-specified improvement rule. Mean development-fold average precision is **0.3905**.

The cutoff **0.22** is selected using development out-of-fold F1. The inference helper applies `probability >= cutoff`, selects the 13 raw predictors in stable order, derives neutral route endpoints, and supports reordered columns and valid unseen categories. The bundle contains the complete pipeline. The native `.cbm` file contains only the classifier and requires the same preprocessing.

The holdout had already been evaluated for the forest before CatBoost development. Its CatBoost results are a secondary check, not an untouched final test. The forest manifest is an unchanged historical record, including statements about its earlier evaluation stage; the selected-model manifest describes the current model. Repeated fold-based selection is optimistic. Identical-input groups do not identify customers, and input availability at scoring time needs confirmation.
