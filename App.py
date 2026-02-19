import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Page Config

st.set_page_config(page_title="LoanSense AI", layout="wide")
st.title("🏦 LoanSense AI System")
st.write("Machine Learning Model to Predict Loan Approval")

# Load Data

data = pd.read_csv("Mini_Project.csv")
data = data.drop("Applicant_ID", axis=1)

# Handle Missing Values

categorical = data.select_dtypes(include=["object"]).columns
numeric = data.select_dtypes(include=["number"]).columns

cat_imp = SimpleImputer(strategy="most_frequent")
data[categorical] = cat_imp.fit_transform(data[categorical])

num_imp = SimpleImputer(strategy="mean")
data[numeric] = num_imp.fit_transform(data[numeric])

# Encoding

onehot_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
               "Property_Area", "Gender", "Employer_Category"]

label_cols = ["Education_Level", "Loan_Approved"]

le = LabelEncoder()
data["Education_Level"] = le.fit_transform(data["Education_Level"])
data["Loan_Approved"] = le.fit_transform(data["Loan_Approved"])

ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

encoded = ohe.fit_transform(data[onehot_cols])
new_cols = ohe.get_feature_names_out(onehot_cols)
encoded_data = pd.DataFrame(encoded, columns=new_cols, index=data.index)

data = pd.concat([data.drop(columns=onehot_cols), encoded_data], axis=1)

# Feature Engineering

data["Credit_Score_sq"] = data["Credit_Score"] ** 2
data["DTI_Ratio_sq"] = data["DTI_Ratio"] ** 2
data["Applicant_Income_log"] = np.log1p(data["Applicant_Income"])

data = data.drop(columns=["Credit_Score", "DTI_Ratio", "Applicant_Income"])

# Train Test Split

X = data.drop("Loan_Approved", axis=1)
Y = data["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model (Logistic Regression)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

acc_score = accuracy_score(y_test, y_pred)
prec_score = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Sidebar Navigation

menu = st.sidebar.selectbox(
    "Navigation",
    ["EDA", "Model Performance", "Predict Loan"]
)

# EDA Section

if menu == "EDA":

    st.subheader("Loan Approval Distribution")

    classes_counts = data["Loan_Approved"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(classes_counts, labels=["No", "Yes"], autopct="%1.1f%%")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    nums_cols = data.select_dtypes("number")
    corr_mat = nums_cols.corr()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_mat, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Model Performance Section

elif menu == "Model Performance":

    st.subheader("Logistic Regression Performance")

    st.write("Accuracy:", round(acc_score, 4))
    st.write("Precision:", round(prec_score, 4))
    st.write("Recall:", round(recall, 4))
    st.write("F1 Score:", round(f1, 4))

    st.write("Confusion Matrix:")
    st.write(cm)

# Prediction Section

elif menu == "Predict Loan":

    st.subheader("Enter Applicant Details")

    age = st.slider("Age", 18, 70, 30)
    savings = st.number_input("Savings", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    credit_score = st.number_input("Credit Score", min_value=300.0)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
    income = st.number_input("Applicant Income", min_value=0.0)

    if st.button("Predict"):

        # Create input dataframe with ORIGINAL columns
        input_dict = {
            "Age": age,
            "Savings": savings,
            "Loan_Amount": loan_amount,
            "Credit_Score": credit_score,
            "DTI_Ratio": dti_ratio,
            "Applicant_Income": income
        }

        input_df = pd.DataFrame([input_dict])

        # Feature Engineering (same as training)
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
        input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
        input_df["Applicant_Income_log"] = np.log1p(input_df["Applicant_Income"])

        input_df = input_df.drop(columns=["Credit_Score", "DTI_Ratio", "Applicant_Income"])

        # Add missing columns (important!)
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[X.columns]

        # Scale
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("Loan Approved ✅")
        else:
            st.error("Loan Not Approved ❌")
