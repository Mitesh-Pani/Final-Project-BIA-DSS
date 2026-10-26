import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="BIA & DSS Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Intelligence & Decision Support System")
st.subheader("Customer Campaign Response Prediction")
st.write(
    "Explore customer behaviour, analyze marketing campaigns, "
    "and evaluate machine learning predictions."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload marketing campaign dataset",
    type=["csv", "tsv", "txt"]
)

if uploaded_file is None:
    st.info("Upload your marketing_campaign.csv file to begin.")
    st.markdown("""
    ### Dashboard features
    - Customer and campaign KPIs
    - Customer demographics and spending analysis
    - Purchase channel comparison
    - Machine learning model evaluation
    - Customer campaign response prediction
    """)
    st.stop()

try:
    df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python"
    )
except Exception as e:
    st.error(f"Unable to read the dataset: {e}")
    st.stop()

df.columns = df.columns.str.strip()

if "Response" not in df.columns:
    st.error(
        "The dataset must contain a 'Response' column "
        "for campaign response analysis."
    )
    st.stop()

df = df.drop_duplicates().copy()

# Clean and prepare data
if "Income" in df.columns:
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Income"] = df["Income"].fillna(df["Income"].median())

if "Year_Birth" in df.columns:
    df["Age"] = pd.Timestamp.now().year - pd.to_numeric(
        df["Year_Birth"], errors="coerce"
    )

children_cols = ["Kidhome", "Teenhome"]
if all(c in df.columns for c in children_cols):
    df["Total_Children"] = df[children_cols].sum(axis=1)

purchase_cols = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases"
]
if all(c in df.columns for c in purchase_cols):
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

spending_cols = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds"
]
available_spending = [c for c in spending_cols if c in df.columns]
if available_spending:
    df["Total_Spending"] = df[available_spending].sum(axis=1)

# Sidebar filters
st.sidebar.header("Dashboard Filters")
filtered_df = df.copy()

if "Education" in df.columns:
    education = st.sidebar.multiselect(
        "Education",
        sorted(df["Education"].dropna().astype(str).unique())
    )
    if education:
        filtered_df = filtered_df[
            filtered_df["Education"].astype(str).isin(education)
        ]

if "Marital_Status" in df.columns:
    status = st.sidebar.multiselect(
        "Marital status",
        sorted(df["Marital_Status"].dropna().astype(str).unique())
    )
    if status:
        filtered_df = filtered_df[
            filtered_df["Marital_Status"].astype(str).isin(status)
        ]

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{len(filtered_df):,}")

if "Income" in filtered_df.columns:
    c2.metric(
        "Average Income",
        f"{filtered_df['Income'].mean():,.0f}"
    )
else:
    c2.metric("Average Income", "N/A")

if "Total_Spending" in filtered_df.columns:
    c3.metric(
        "Average Spending",
        f"{filtered_df['Total_Spending'].mean():,.0f}"
    )
else:
    c3.metric("Average Spending", "N/A")

response = pd.to_numeric(filtered_df["Response"], errors="coerce")
response_rate = response.mean() * 100 if response.notna().any() else 0
c4.metric("Response Rate", f"{response_rate:.1f}%")

st.divider()

# Dataset overview
st.header("1. Customer Data Overview")
st.dataframe(filtered_df.head(100), use_container_width=True)
st.caption("Showing up to 100 rows of the filtered dataset.")

# Response distribution
st.header("2. Campaign Response Analysis")

response_counts = (
    filtered_df["Response"]
    .astype(str)
    .value_counts()
    .rename_axis("Response")
    .reset_index(name="Customers")
)

fig_response = px.bar(
    response_counts,
    x="Response",
    y="Customers",
    color="Response",
    title="Customer Campaign Response Distribution"
)
st.plotly_chart(fig_response, use_container_width=True)

left, right = st.columns(2)

with left:
    if "Education" in filtered_df.columns:
        edu_data = (
            filtered_df.groupby(
                ["Education", "Response"], observed=True
            ).size().reset_index(name="Customers")
        )
        fig = px.bar(
            edu_data,
            x="Education",
            y="Customers",
            color="Response",
            barmode="group",
            title="Campaign Response by Education"
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    if "Marital_Status" in filtered_df.columns:
        marital_data = (
            filtered_df.groupby(
                ["Marital_Status", "Response"], observed=True
            ).size().reset_index(name="Customers")
        )
        fig = px.bar(
            marital_data,
            x="Marital_Status",
            y="Customers",
            color="Response",
            barmode="group",
            title="Campaign Response by Marital Status"
        )
        st.plotly_chart(fig, use_container_width=True)

# Spending and channel analysis
st.header("3. Spending and Purchase Analysis")

left, right = st.columns(2)

with left:
    if "Total_Spending" in filtered_df.columns:
        fig = px.histogram(
            filtered_df,
            x="Total_Spending",
            nbins=30,
            title="Customer Spending Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    if all(c in filtered_df.columns for c in purchase_cols):
        channel_data = pd.DataFrame({
            "Channel": ["Web", "Catalog", "Store"],
            "Purchases": [
                filtered_df["NumWebPurchases"].sum(),
                filtered_df["NumCatalogPurchases"].sum(),
                filtered_df["NumStorePurchases"].sum()
            ]
        })
        fig = px.bar(
            channel_data,
            x="Channel",
            y="Purchases",
            title="Purchases by Channel"
        )
        st.plotly_chart(fig, use_container_width=True)

# Machine learning
st.header("4. Machine Learning Model Evaluation")

st.write(
    "Train a Random Forest model to predict campaign response "
    "using the uploaded dataset."
)

if st.button("Train and Evaluate Model", type="primary"):
    try:
        model_df = df.copy()

        # The target must be known for training.
        model_df["Response"] = pd.to_numeric(
            model_df["Response"], errors="coerce"
        )
        model_df = model_df.dropna(subset=["Response"])

        target_values = set(model_df["Response"].unique())
        if not target_values.issubset({0, 1}) or len(target_values) != 2:
            st.error(
                "Response must contain two classes encoded as 0 and 1."
            )
            st.stop()

        drop_cols = [
            "Response", "ID", "Dt_Customer",
            "Z_CostContact", "Z_Revenue"
        ]

        X = model_df.drop(
            columns=[c for c in drop_cols if c in model_df.columns]
        )
        y = model_df["Response"].astype(int)

        # Remove columns that are completely empty.
        X = X.dropna(axis=1, how="all")

        # Remove identifiers and columns with too many unique values.
        remove_cols = []
        for col in X.columns:
            if (
                X[col].nunique(dropna=True) == len(X)
                or X[col].nunique(dropna=True) > 0.95 * len(X)
            ):
                remove_cols.append(col)

        X = X.drop(columns=remove_cols)

        if X.shape[1] == 0:
            st.error("No usable feature columns remain.")
            st.stop()

        numeric_cols = X.select_dtypes(
            include=np.number
        ).columns.tolist()
        categorical_cols = X.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", numeric_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(
                ("categorical", categorical_pipe, categorical_cols)
            )

        preprocessor = ColumnTransformer(transformers)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100
        precision = precision_score(
            y_test, predictions, zero_division=0
        ) * 100
        recall = recall_score(
            y_test, predictions, zero_division=0
        ) * 100
        f1 = f1_score(
            y_test, predictions, zero_division=0
        ) * 100

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{accuracy:.2f}%")
        b.metric("Precision", f"{precision:.2f}%")
        c.metric("Recall", f"{recall:.2f}%")
        d.metric("F1-score", f"{f1:.2f}%")

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual No Response", "Actual Response"],
            columns=["Predicted No Response", "Predicted Response"]
        )

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

        st.success(
            "Model trained successfully. These metrics are from the "
            "current dataset and train/test split; they are not guaranteed "
            "to match results from your notebook."
        )

    except Exception as e:
        st.error(f"Model training failed: {e}")

st.divider()
st.caption("BIA & DSS | Predictive Analytics for Customer Campaign Response")
