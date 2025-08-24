# Heart Disease Prediction (UCI) — Full Pipeline

This project trains a classifier for Heart Disease using the official UCI repository via `ucimlrepo` (dataset id=45).
- First run downloads the dataset programmatically.
- Trained pipeline is saved at `models/final_model.pkl`.
- Metrics are written to `results/evaluation_metrics.txt`.
- A Streamlit app in `ui/app.py` allows interactive predictions and on-demand training.

## Quickstart

```bash
python -m venv venv
# mac/linux
source venv/bin/activate
# windows
venv\Scripts\activate

pip install -r requirements.txt

# Train (downloads the dataset using ucimlrepo on first run)
python train.py

# Run UI
cd ui
streamlit run app.py
```