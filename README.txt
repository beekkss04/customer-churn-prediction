CUSTOMER CHURN COMPLETE DASHBOARD

Keep these files in the same folder:
- app.py
- E Commerce Dataset.xlsx
- customer_churn_model.pkl
- customer_features.pkl
- requirements.txt

MAC INSTRUCTIONS
Use Python 3.11 for the most predictable compatibility with the exported scikit-learn model.
From Terminal, inside this folder:

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py

If python3.11 is not installed, install Python 3.11 first from python.org, then run the commands above.

The sidebar contains project overview, dataset and data quality, EDA, preprocessing,
model comparison, model evaluation, feature importance, live prediction, and conclusion.

NOTE
Model metrics and the confusion matrix are the previously recorded notebook results.
The EDA charts are calculated directly from the supplied Excel workbook.
Feature importance is displayed as the recorded rank order only; exact numerical
permutation-importance values were not supplied in the exported files.
