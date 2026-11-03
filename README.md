# British Airways Data Science Job Simulation

Completed work for the British Airways Data Science simulation on [Forage](https://www.theforage.com/virtual-experience/NjynCWzGSaWXQCxSX/british-airways/data-science-yqoz). The two tasks cover lounge eligibility planning in Excel and customer booking prediction in Python.

This is an educational portfolio project using simulation data, not employment at British Airways or a production airline system.

## Explore the work

| Task | Question | Main deliverables |
| --- | --- | --- |
| [1 · Lounge eligibility](task1/) | How can lounge eligibility be estimated for future flight schedules? | [Excel model](task1/BA%20Lounge%20Eligibility%20Model.xlsx) · [Completed submission](task1/submission/BA%20Lounge%20Eligibility%20Submission.xlsx) |
| [2 · Booking prediction](task2/) | Which booking records are more likely to result in a completed purchase? | [Executed notebook](task2/Booking%20Analysis%20-%20Recommended%20Changes.ipynb) · [One-slide summary](task2/submission/Booking%20Behaviour%20Summary.pptx) · [Results](task2/results/verified_results.json) |

## Task 1: Lounge eligibility planning

The workbook groups a 10,000-flight schedule into eight categories: short/long haul crossed with four times of day. It calculates eligibility rates for three lounge tiers and applies the lookup to a sample of flights. The completed submission explains the grouping, assumptions and application to future schedules.

Rates use total eligible passengers divided by total seats within each group. Seats serve as a passenger proxy under a full-occupancy assumption. These are eligibility estimates, not forecasts of visits or simultaneous occupancy. The simulation treats Tier 1 as hypothetical future capacity at Heathrow Terminal 3.

**Tools:** Excel/Google Sheets, formulas, grouped aggregation and lookup modelling.

## Task 2: Customer booking prediction

The analysis uses 50,000 booking records, of which 14.96% have a completed booking. Identical input patterns stay together across the development/holdout split and five cross-validation folds. Preprocessing is fitted within each fold.

The notebook compares logistic regression, random forest and CatBoost. CatBoost is the selected model; its decision threshold of **0.22** maximises F1 on development out-of-fold predictions.

| Model | Mean development-fold average precision | Mean development-fold ROC-AUC |
| --- | ---: | ---: |
| Logistic regression | 0.3672 | 0.7890 |
| Random forest | 0.3857 | 0.7943 |
| Selected CatBoost | **0.3905** | **0.7998** |

On the holdout, CatBoost achieves average precision **0.3735**, ROC-AUC **0.7882**, precision **32.9%** and recall **64.1%**. The notebook documents the evaluation procedure and its limitations.

Permutation importance identifies route and booking origin as the strongest predictive inputs. These associations are not causal. Without customer IDs or a dated booking sequence, the grouped split cannot establish performance on new customers or future periods. Input availability before booking completion also needs confirmation before any operational use.

**Tools:** Python, pandas, NumPy, scikit-learn, CatBoost, Matplotlib and Jupyter.

## Repository contents

```text
task1/
  BA Lounge Eligibility Model.xlsx
  data/                  Forage flight schedule
  resources/             Assignment notes and original submission template
  submission/            Completed Excel submission
task2/
  Booking Analysis - Recommended Changes.ipynb
  booking_model.py        Input validation and selected-model inference
  requirements*.txt      Pinned dependencies and recorded environment
  data/                  Forage booking dataset
  resources/             Starter notebook and assignment notes
  model/                 Selected CatBoost and historical forest benchmark
  results/               Verified metrics and feature importance
  submission/            Editable one-slide PowerPoint summary
```

## Run the analysis

The recorded environment is **Python 3.13.16 on macOS**. From the repository root:

```bash
python3.13 -m venv task2/.venv
source task2/.venv/bin/activate
python -m pip install -r task2/requirements.txt
python -m ipykernel install --user --name ba-forage --display-name "Python (BA Forage)"
```

Open the Task 2 notebook in a Jupyter-compatible editor, select **Python (BA Forage)**, and run all cells from the repository root or `task2/`. A full run takes several minutes and regenerates the selected model and results. The notebook checks dependency versions, dataset hashes and historical benchmark hashes before fitting. `requirements-lock.txt` records the full original environment; the core requirements are sufficient for the analysis.

For inference without retraining, run this from `task2/` in the same environment:

```python
import pandas as pd
from booking_model import predict_completed

rows = pd.read_csv("data/customer_booking.csv", encoding="ISO-8859-1")
predictions = predict_completed(rows.drop(columns="booking_complete").head())
print(predictions)
```

The helper checks the selected bundle's versions and hashes. Only load pickle artifacts from a source you trust. The native `.cbm` classifier requires the same feature preparation as the Python pipeline.

## Sources and attribution

British Airways and Forage supplied the task briefs, datasets, Task 1 submission template and Task 2 starter notebook. They are included with the completed work for this educational simulation. Original source links and task requirements are recorded in each task's `resources/` folder. The supplied materials retain their original ownership and simulation-use designation; this repository does not grant a separate licence to those materials.

The completed workbook, analysis, inference helper and presentation document the project work. AI assistance was used during development, review and documentation.
