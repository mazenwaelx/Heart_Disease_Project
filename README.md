
# Heart Disease UCI — Comprehensive ML Pipeline

- Preprocessing, PCA, Feature Selection (RF, RFE, Chi-square)
- Supervised models (LogReg, DT, RF, SVM) + metrics
- Hyperparameter tuning (GridSearchCV via notebook 06)
- Unsupervised (KMeans)
- Streamlit UI with first-run training
- Ngrok for sharing

## Run
```bash
python -m venv venv
# mac/linux
source venv/bin/activate
# windows
venv\Scripts\activate

pip install -r requirements.txt
cd ui
streamlit run app.py
```
