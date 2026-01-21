import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/attrition_model.pkl")

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.title("👔 Employee Attrition Prediction")
st.write(
    """
    This system predicts **employee attrition risk** based on historical HR data.

    ⚠️ The prediction is **probability-based** and optimized to identify
    employees at risk of leaving (recall-focused).
    """
)

st.divider()

# -----------------------------
# USER INPUTS (same as before)
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    distance = st.slider("Distance From Home", 1, 50, 10)

with col2:
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Research Director", "Human Resources"
        ]
    )
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    business_travel = st.selectbox(
        "Business Travel",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    )

with col3:
    monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
    percent_hike = st.slider("Percent Salary Hike", 10, 30, 15)
    stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

st.divider()

# -----------------------------
# SATISFACTION & EXPERIENCE
# -----------------------------
job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
environment_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
relationship_satisfaction = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
work_life_balance = st.slider("Work-Life Balance (1–4)", 1, 4, 3)

total_work_years = st.slider("Total Working Years", 0, 40, 10)
years_at_company = st.slider("Years at Company", 0, 40, 5)
years_in_role = st.slider("Years in Current Role", 0, 20, 3)
years_with_manager = st.slider("Years With Current Manager", 0, 20, 3)
years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
num_companies = st.slider("Number of Companies Worked", 0, 10, 2)

overtime = st.selectbox("OverTime", ["Yes", "No"])

# -----------------------------
# INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital,
    "DistanceFromHome": distance,
    "Education": 3,
    "EducationField": "Life Sciences",

    "Department": department,
    "JobRole": job_role,
    "JobLevel": job_level,
    "JobInvolvement": 3,
    "BusinessTravel": business_travel,

    "MonthlyIncome": monthly_income,
    "MonthlyRate": 15000,
    "DailyRate": 800,
    "HourlyRate": 60,
    "PercentSalaryHike": percent_hike,
    "StockOptionLevel": stock_option,

    "TotalWorkingYears": total_work_years,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_role,
    "YearsWithCurrManager": years_with_manager,
    "YearsSinceLastPromotion": years_since_promo,
    "NumCompaniesWorked": num_companies,

    "JobSatisfaction": job_satisfaction,
    "EnvironmentSatisfaction": environment_satisfaction,
    "RelationshipSatisfaction": relationship_satisfaction,
    "WorkLifeBalance": work_life_balance,

    "OverTime": overtime,
    "Over18": "Y",
    "EmployeeCount": 1,
    "StandardHours": 80,
    "PerformanceRating": 3,
    "TrainingTimesLastYear": 2,
    "EmployeeNumber": 9999
}])

# -----------------------------
# PREDICTION WITH CUSTOM THRESHOLD
# -----------------------------
THRESHOLD = 0.30  # recall-optimized

if st.button("Predict Attrition Risk"):
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= THRESHOLD else 0

    if prediction == 1:
        st.error(
            f"⚠️ **High Attrition Risk Detected**\n\n"
            f"Probability of leaving: **{proba * 100:.1f}%**\n\n"
            f"(Threshold used: {THRESHOLD})"
        )
        st.warning(
            "Recommended actions:\n"
            "- Reduce overtime\n"
            "- Improve work-life balance\n"
            "- Review compensation & growth opportunities"
        )
    else:
        st.success(
            f"✅ **Lower Attrition Risk**\n\n"
            f"Probability of leaving: **{proba * 100:.1f}%**"
        )
        st.info(
            "Employee shows relatively stable retention indicators, "
            "but continued engagement is advised."
        )
