Heart Disease Prediction Project

A comprehensive machine learning pipeline for analyzing and predicting heart disease risk using the UCI Heart Disease Dataset. The project covers data preprocessing, feature selection, dimensionality reduction (PCA), supervised and unsupervised modeling, hyperparameter tuning, and deployment through a Streamlit UI.

Features
1. Data Preprocessing & Cleaning

Missing values handled via imputation.

Categorical encoding with One-Hot Encoder.

Feature scaling using StandardScaler/MinMaxScaler.

Exploratory Data Analysis (EDA) with histograms, correlation heatmaps, and boxplots.

2. Dimensionality Reduction

Principal Component Analysis (PCA) applied.

Cumulative variance plots to determine optimal components.

3. Feature Selection

Random Forest feature importance ranking.

Recursive Feature Elimination (RFE).

Chi-Square test for categorical features.

4. Supervised Learning

Trained and evaluated with multiple models:

Logistic Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

Metrics include: Accuracy, Precision, Recall, F1-Score, ROC curve & AUC.

5. Unsupervised Learning

K-Means clustering with elbow method.

Hierarchical clustering with dendrograms.

6. Hyperparameter Tuning

Implemented with GridSearchCV (with example for RandomizedSearchCV).

Optimized model saved to .pkl.

7. Deployment

Streamlit UI for real-time prediction.

Ngrok setup for public deployment.

Results logged into results/evaluation_metrics.txt.

Technical Stack

Python 3.9+

Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, ucimlrepo, joblib, scipy

Deployment: Streamlit + Ngrok

Version Control: Git & GitHub

File Structure
Heart_Disease_Project/
│── data/
│   └── heart_disease.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│── models/
│   └── final_model.pkl
│── ui/
│   └── app.py
│── deployment/
│   └── ngrok_setup.txt
│── results/
│   └── evaluation_metrics.txt
│── README.md
│── requirements.txt
└── .gitignore

Getting Started
Prerequisites

Python 3.9+

pip / virtualenv

Installation
# clone the repo
git clone https://github.com/mazenwaelx/Heart_Disease_Project.git

# move into project directory
cd Heart_Disease_Project

# create environment & install dependencies
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

Run the app
cd ui
streamlit run app.py


The UI will launch locally at http://localhost:8501
.

Contributing

Fork the repository.

Create a new branch (git checkout -b feature-branch).

Commit your changes (git commit -am 'Add feature').

Push to branch (git push origin feature-branch).

Open a Pull Request.

License

This project is for educational and research purposes. Not intended for clinical use.