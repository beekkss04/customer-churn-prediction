import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Customer Churn Analytics", page_icon="🛍️", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "E Commerce Dataset.xlsx"
MODEL_PATH = ROOT / "customer_churn_model.pkl"
FEATURES_PATH = ROOT / "customer_features.pkl"

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 14px; border-radius: 12px;}
div[data-testid="stForm"] {border: 1px solid rgba(128,128,128,.25); padding: 18px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH, sheet_name="E Comm")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)

if not DATA_PATH.exists():
    st.error("Dataset file is missing. Keep E Commerce Dataset.xlsx beside app.py.")
    st.stop()

df = load_data()
churn_rate = float(df["Churn"].mean()) if "Churn" in df else np.nan
total = len(df)
churned = int((df["Churn"] == 1).sum())
retained = int((df["Churn"] == 0).sum())
missing_total = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

st.sidebar.title("🛍️ Churn Analytics")
st.sidebar.caption("E-Commerce • Machine Learning Mini Project")
page = st.sidebar.radio("NAVIGATION", [
    "Project overview", "Dataset & data quality", "Exploratory data analysis",
    "Preprocessing workflow", "Model comparison", "Model evaluation",
    "Feature importance", "Live churn prediction", "Conclusion & future scope"
])
st.sidebar.divider()
st.sidebar.caption("Academic prototype • Historical churn label")

def page_title(title, subtitle):
    st.title(title)
    st.caption(subtitle)

def chart_note(note):
    st.caption("How to read this: " + note)

if page == "Project overview":
    page_title("Customer Churn Prediction", "A supervised machine-learning project for an e-commerce customer dataset.")
    st.markdown("This dashboard brings the dataset, exploratory analysis, preprocessing workflow, model results and interactive prediction into one place.")
    a,b,c,d = st.columns(4)
    a.metric("Customer records", f"{total:,}")
    b.metric("Churned customers", f"{churned:,}")
    c.metric("Retained customers", f"{retained:,}")
    d.metric("Churn rate", f"{churn_rate:.1%}")
    st.subheader("Project objective")
    st.write("Predict whether a customer is marked as churned (Churn = 1) or retained (Churn = 0) using historical customer and shopping-behaviour information.")
    st.subheader("Project workflow")
    steps = ["Problem definition", "Dataset understanding", "Data cleaning & preprocessing", "Exploratory data analysis", "Train three classifiers", "Evaluate & compare", "Interpret features", "Predict a customer"]
    cols = st.columns(4)
    for i, step in enumerate(steps):
        cols[i % 4].info(f"{i+1}. {step}")
    st.subheader("Model snapshot")
    st.write("The notebook's recorded test-split results are shown below. They are not a guarantee of future performance.")
    metrics = pd.DataFrame([
        ["Random Forest", .9742, .9939, .8526, .9178],
        ["Decision Tree", .8357, .5080, .8368, .6322],
        ["Logistic Regression", .7904, .4372, .8421, .5755],
    ], columns=["Model","Accuracy","Precision","Recall","F1-score"])
    st.dataframe(metrics.style.format({c:"{:.2%}" for c in metrics.columns[1:]}), use_container_width=True, hide_index=True)

elif page == "Dataset & data quality":
    page_title("Dataset & data quality", "Inspect the supplied E Comm worksheet and its completeness.")
    a,b,c,d = st.columns(4)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", df.shape[1])
    c.metric("Missing cells", f"{missing_total:,}")
    d.metric("Exact duplicate rows", f"{duplicate_rows:,}")
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Missing values by column")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        fig = px.bar(x=missing.values, y=missing.index, orientation="h", labels={"x":"Missing values","y":"Column"}, title="Missing values in each column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found.")
    st.subheader("Column types")
    types = pd.DataFrame({"Column":df.columns, "Data type":[str(t) for t in df.dtypes], "Missing":df.isna().sum().values, "Unique values":df.nunique(dropna=True).values})
    st.dataframe(types, use_container_width=True, hide_index=True)
    st.info("CustomerID is an identifier, not a behavioural predictor. Churn is the target label.")

elif page == "Exploratory data analysis":
    page_title("Exploratory data analysis", "Charts are generated directly from the uploaded E Comm worksheet.")
    st.subheader("1. Churn class distribution")
    class_df = df["Churn"].value_counts().rename_axis("Churn label").reset_index(name="Customers")
    class_df["Customer status"] = class_df["Churn label"].map({0:"Retained (0)",1:"Churned (1)"})
    fig = px.pie(class_df, names="Customer status", values="Customers", hole=.48, title="Retained vs churned customers")
    st.plotly_chart(fig, use_container_width=True)
    chart_note("Shows the share of customers assigned each historical target label.")
    st.subheader("2. Tenure distribution by churn")
    fig = px.histogram(df, x="Tenure", color="Churn", marginal="box", barmode="overlay", opacity=.65, nbins=30, title="Customer tenure and churn label")
    st.plotly_chart(fig, use_container_width=True)
    chart_note("Compare tenure distributions between retained and churn-labelled customers; this is an association, not proof of cause.")
    st.subheader("3. Churn by complaint status")
    comp = df.groupby("Complain", dropna=False)["Churn"].mean().reset_index()
    comp["Complaint status"] = comp["Complain"].map({0:"No complaint",1:"Complaint"})
    comp["Churn rate"] = comp["Churn"] * 100
    fig = px.bar(comp, x="Complaint status", y="Churn rate", text_auto=".1f", title="Historical churn rate by complaint status", labels={"Churn rate":"Churn rate (%)"})
    st.plotly_chart(fig, use_container_width=True)
    chart_note("The chart compares the proportion labelled churned in each complaint group.")
    st.subheader("4. Preferred order category")
    cat = df.groupby("PreferedOrderCat", dropna=False)["Churn"].mean().mul(100).reset_index(name="Churn rate (%)").sort_values("Churn rate (%)", ascending=False)
    fig = px.bar(cat, x="PreferedOrderCat", y="Churn rate (%)", title="Churn rate by preferred order category")
    st.plotly_chart(fig, use_container_width=True)
    chart_note("Categories with higher historical churn rates may deserve closer investigation.")
    st.subheader("5. Satisfaction score")
    sat = df.groupby("SatisfactionScore", dropna=False)["Churn"].mean().mul(100).reset_index(name="Churn rate (%)")
    fig = px.line(sat, x="SatisfactionScore", y="Churn rate (%)", markers=True, title="Churn rate across satisfaction scores")
    st.plotly_chart(fig, use_container_width=True)
    chart_note("This shows how churn-label proportion varies across recorded satisfaction scores.")
    st.subheader("6. Cashback amount distribution")
    fig = px.box(df, x="Churn", y="CashbackAmount", color="Churn", title="Cashback amount by churn label", labels={"Churn":"Churn label"})
    st.plotly_chart(fig, use_container_width=True)
    chart_note("Box plots show median, spread and outliers for cashback amount in each class.")
    st.subheader("7. Numeric feature correlation")
    numeric = df.select_dtypes(include=np.number).drop(columns=["CustomerID"], errors="ignore")
    corr = numeric.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap of numeric columns")
    st.plotly_chart(fig, use_container_width=True)
    chart_note("Correlation measures pairwise linear association; it does not establish causation.")

elif page == "Preprocessing workflow":
    page_title("Preprocessing workflow", "The notebook's intended preparation pipeline, before fitting classifiers.")
    workflow = [
        ("1. Remove exact duplicates", "Avoid counting identical records more than once."),
        ("2. Handle missing target", "Rows without a known Churn label cannot be used as supervised examples."),
        ("3. Exclude CustomerID", "An identifier is not used as a behavioural predictor."),
        ("4. Impute numeric columns", "Median imputation fills missing numeric values."),
        ("5. Impute categorical columns", "Most-frequent imputation fills missing category values."),
        ("6. One-hot encode categories", "Convert categorical values into model-readable indicator columns."),
        ("7. Standardize numeric values", "Scale numeric features for models sensitive to feature scale."),
        ("8. Stratified train/test split", "Keep the class proportion similar across the 80:20 split."),
        ("9. Fit preprocessing in a pipeline", "Learn transformations from training data to reduce data leakage.")
    ]
    for title, desc in workflow:
        with st.expander(title, expanded=True):
            st.write(desc)
    st.warning("This page describes the notebook workflow. Confirm the executed notebook cells and outputs when preparing the final academic submission.")

elif page == "Model comparison":
    page_title("Model comparison", "Three supervised classification algorithms evaluated on the notebook's held-out test split.")
    results = pd.DataFrame([
        ["Random Forest", .9742, .9939, .8526, .9178],
        ["Decision Tree", .8357, .5080, .8368, .6322],
        ["Logistic Regression", .7904, .4372, .8421, .5755],
    ], columns=["Model","Accuracy","Precision","Recall","F1-score"])
    melted = results.melt("Model", var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", range_y=[0,1], text_auto=".1%", title="Model performance comparison")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results.style.format({c:"{:.2%}" for c in results.columns[1:]}), use_container_width=True, hide_index=True)
    st.markdown("**Recorded best model by F1-score: Random Forest.** F1 balances precision and recall; the preferred metric depends on the business cost of missed churners versus unnecessary retention actions.")
    st.caption("These are the previously recorded results from the notebook run. Re-run evaluation in Colab if you change the data split, preprocessing or model settings.")

elif page == "Model evaluation":
    page_title("Model evaluation", "Inspect the recorded Random Forest test-set confusion matrix and metrics.")
    cm = np.array([[935,1],[28,162]])
    fig = px.imshow(cm, text_auto=True, x=["Predicted retained","Predicted churned"], y=["Actual retained","Actual churned"], color_continuous_scale="Blues", title="Random Forest confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    a,b,c,d = st.columns(4)
    a.metric("Accuracy", "97.42%")
    b.metric("Precision", "99.39%")
    c.metric("Recall", "85.26%")
    d.metric("F1-score", "91.78%")
    st.markdown("""
    - **True negatives (935):** retained customers correctly predicted retained.
    - **False positives (1):** retained customers incorrectly predicted churned.
    - **False negatives (28):** churn-labelled customers predicted retained.
    - **True positives (162):** churn-labelled customers correctly predicted churned.
    """)
    st.warning("The model missed 28 of 190 churn-labelled customers in this test split. High accuracy alone does not mean every churner is detected.")

elif page == "Feature importance":
    page_title("Feature importance", "Features ranked by the notebook's recorded permutation-importance output.")
    ranking = ["Tenure","Complain","MaritalStatus","PreferedOrderCat","CashbackAmount","DaySinceLastOrder","NumberOfAddress","PreferredPaymentMode","WarehouseToHome","SatisfactionScore"]
    rank_df = pd.DataFrame({"Feature":ranking, "Rank":range(1,len(ranking)+1)})
    fig = px.bar(rank_df.sort_values("Rank", ascending=True), x="Rank", y="Feature", orientation="h", title="Top predictive features (rank order)")
    fig.update_layout(yaxis={"categoryorder":"total ascending"}, xaxis_title="Rank (1 = highest)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)
    st.info("This is the recorded feature ranking, not numerical permutation-importance scores. Permutation importance measures predictive usefulness in this model; it does not prove that a feature causes churn.")

elif page == "Live churn prediction":
    page_title("Live customer churn prediction", "Enter a customer profile and run the saved Random Forest pipeline.")
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error("Model files are missing. Keep both .pkl files beside app.py.")
        st.stop()
    try:
        model, feature_names = load_model()
    except Exception as e:
        st.error("Could not load the saved model. Use the scikit-learn version from the Colab export environment.")
        st.code(str(e))
        st.stop()
    with st.form("predict_form"):
        st.subheader("Customer profile")
        c1,c2,c3 = st.columns(3)
        with c1:
            tenure=st.number_input("Tenure (months)",0.0,100.0,17.0)
            login=st.selectbox("Preferred login device",sorted(df["PreferredLoginDevice"].dropna().unique()))
            city=st.selectbox("City tier",sorted(df["CityTier"].dropna().unique().tolist()))
            wh=st.number_input("Warehouse to home",0.0,200.0,9.0)
            payment=st.selectbox("Preferred payment mode",sorted(df["PreferredPaymentMode"].dropna().unique()))
            gender=st.selectbox("Gender",sorted(df["Gender"].dropna().unique()))
        with c2:
            hours=st.number_input("Hours spent on app",0.0,24.0,4.0)
            devices=st.number_input("Registered devices",0,20,4)
            category=st.selectbox("Preferred order category",sorted(df["PreferedOrderCat"].dropna().unique()))
            satisfaction=st.select_slider("Satisfaction score",options=sorted(df["SatisfactionScore"].dropna().unique().tolist()),value=1)
            marital=st.selectbox("Marital status",sorted(df["MaritalStatus"].dropna().unique()))
            addresses=st.number_input("Number of addresses",1,50,3)
        with c3:
            complain=st.selectbox("Has customer complained?",[0,1],format_func=lambda v:"Yes" if v else "No")
            hike=st.number_input("Order amount hike from last year",0.0,200.0,21.0)
            coupons=st.number_input("Coupons used",0,100,6)
            orders=st.number_input("Order count",0,100,8)
            days=st.number_input("Days since last order",0.0,365.0,8.0)
            cashback=st.number_input("Cashback amount",0.0,10000.0,181.75)
        go=st.form_submit_button("Predict customer churn", type="primary", use_container_width=True)
    if go:
        record={"Tenure":tenure,"PreferredLoginDevice":login,"CityTier":city,"WarehouseToHome":wh,
        "PreferredPaymentMode":payment,"Gender":gender,"HourSpendOnApp":hours,"NumberOfDeviceRegistered":devices,
        "PreferedOrderCat":category,"SatisfactionScore":satisfaction,"MaritalStatus":marital,"NumberOfAddress":addresses,
        "Complain":complain,"OrderAmountHikeFromlastYear":hike,"CouponUsed":coupons,"OrderCount":orders,
        "DaySinceLastOrder":days,"CashbackAmount":cashback}
        inp=pd.DataFrame([[record.get(f) for f in feature_names]],columns=feature_names)
        try:
            pred=int(model.predict(inp)[0])
            probs=model.predict_proba(inp)[0] if hasattr(model,"predict_proba") else None
            churn_prob=float(probs[list(model.classes_).index(1)]) if probs is not None and 1 in list(model.classes_) else None
            st.divider()
            if pred==1: st.error("⚠️ Model prediction: Churned (1)")
            else: st.success("✅ Model prediction: Retained (0)")
            if churn_prob is not None:
                st.metric("Estimated churn probability",f"{churn_prob:.2%}")
                st.progress(float(np.clip(churn_prob,0,1)))
            st.caption("This prediction reflects patterns learned from the historical dataset; it is not a certainty or a causal explanation.")
        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))

elif page == "Conclusion & future scope":
    page_title("Conclusion, limitations & future scope", "A concise summary of what this educational model can and cannot establish.")
    st.subheader("Conclusion")
    st.write("The project applies supervised binary classification to predict the historical churn label in an e-commerce dataset. Three algorithms were compared, and Random Forest recorded the strongest F1-score in the results shown in this dashboard.")
    st.subheader("Limitations")
    limitations = [
        "The model learns from the supplied dataset and may not generalize to other companies or time periods.",
        "The exact business definition and observation window for the source Churn label are not fully documented in the data dictionary.",
        "Historical associations do not establish that a feature causes churn.",
        "Class imbalance means accuracy should be considered alongside precision, recall and F1-score.",
        "This is an academic prototype and has not been validated in a live business environment."
    ]
    for item in limitations: st.markdown(f"- {item}")
    st.subheader("Future scope")
    for item in ["Test on a newer time-based dataset and monitor performance over time.","Tune model hyperparameters and compare additional algorithms.","Calibrate predicted probabilities and choose a business-appropriate decision threshold.","Connect the model to a secure customer database and monitor drift.","Design and evaluate retention interventions through controlled experiments."]:
        st.markdown(f"- {item}")
    st.caption("Use the actual executed notebook outputs and your institution's report format when preparing the final submission.")
