import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# Remove default top margin
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("Dashboard - Understanding the Global Burden and Risk Factors of Heart Disease")
    st.caption("MSBA 382 - Individual Project")
    st.header("Filters")

# Load preprocessed datasets 
expanded_df = pd.read_csv("Healthcare/expanded_daly.csv")
df_deaths = pd.read_csv("Healthcare/expanded_top_causes.csv")
cardio_data = pd.read_csv("Healthcare/cardio_transformed.csv")


# Filters
age_min, age_max = st.sidebar.slider("Select Age Range (in years):", 0, 100, (0, 100))
gender = st.sidebar.radio("Select Gender:", ["Both", "Male", "Female"])

if gender != "Both":
    expanded_df = expanded_df[expanded_df["Sex"] == gender]

filtered_df = expanded_df[
    (expanded_df["NumericAge"] >= age_min) &
    (expanded_df["NumericAge"] <= age_max) &
    (expanded_df["Cause of death or injury"] == "Cardiovascular diseases") &
    (expanded_df["Measure"] == "DALYs per 100,000")
]

import joblib

# Load trained model
model = joblib.load("Healthcare/logreg_model.joblib")

# Sidebar Inputs for Prediction
st.sidebar.title("")
st.sidebar.header("Predict Risk Based on Your Health Profile")

age_input = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=45)
gender_input = st.sidebar.radio("Gender", ["Male", "Female"])
systolic = st.sidebar.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120)
diastolic = st.sidebar.number_input("Diastolic Blood Pressure", min_value=50, max_value=150, value=80)
chol_input = st.sidebar.selectbox("LDL Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
gluc_input = st.sidebar.selectbox("Fasting Glucose", ["Normal", "Above Normal", "Well Above Normal"])
smoke_input = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
alco_input = st.sidebar.selectbox("Alcohol Intake?", ["No", "Yes"])
active_input = st.sidebar.selectbox("Physically Active?", ["Yes", "No"])
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)

# Compute Derived Fields
bmi = weight / ((height / 100) ** 2)

# Convert to encoded format used during training
gender_input_encoded = 2 if gender_input == "Male" else 1
chol_encoded = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}[chol_input]
gluc_encoded = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}[gluc_input]
smoke = 1 if smoke_input == "Yes" else 0
alco = 1 if alco_input == "Yes" else 0
active = 1 if active_input == "Yes" else 0

# Age group one-hot
age_group = "<40" if age_input < 40 else "40-49" if age_input < 50 else "50-59" if age_input < 60 else "60+"
age_groups = {"age_group_40-49": 0, "age_group_50-59": 0, "age_group_60+": 0}
if age_group in ["40-49", "50-59", "60+"]:
    age_groups[f"age_group_{age_group}"] = 1

# Manual one-hot for cholesterol and glucose
chol_cols = {"cholesterol_2": 0, "cholesterol_3": 0}
if chol_encoded == 1:
    chol_cols["cholesterol_2"] = 1
elif chol_encoded == 2:
    chol_cols["cholesterol_3"] = 1

gluc_cols = {"gluc_2": 0, "gluc_3": 0}
if gluc_encoded == 1:
    gluc_cols["gluc_2"] = 1
elif gluc_encoded == 2:
    gluc_cols["gluc_3"] = 1

# Final feature vector
# Final feature vector
input_df = pd.DataFrame([{
    "gender": gender_input_encoded,
    "ap_hi": systolic,
    "ap_lo": diastolic,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "BMI": bmi,
    **age_groups,
    **chol_cols,
    **gluc_cols
}])


# Predict
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

# Output
st.sidebar.subheader("Prediction Result")
if prediction == 1:
    st.sidebar.error("High Risk of Heart Disease")
else:
    st.sidebar.success("Low Risk of Heart Disease")


# Column layout with narrow separator
col1, col_sep, col3 = st.columns([2.2, 0.1, 1.5])

# --- COLUMN 1 ---
with col1:
    st.markdown("<h5 style='margin-bottom: 10px;'>Heart Disease Among the Leading Causes of Death by Gender</h5>", unsafe_allow_html=True)

    filtered_deaths = df_deaths[(df_deaths["NumericAge"] >= age_min) & (df_deaths["NumericAge"] <= age_max)]

    female_top = filtered_deaths[filtered_deaths["Gender"] == "Female"].groupby("DIM_GHECAUSE_TITLE")["VAL_DTHS_RATE100K_NUMERIC"].sum()
    male_top = filtered_deaths[filtered_deaths["Gender"] == "Male"].groupby("DIM_GHECAUSE_TITLE")["VAL_DTHS_RATE100K_NUMERIC"].sum()
    combined = (female_top + male_top).sort_values(ascending=False).head(10)
    common_causes = combined.index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=common_causes,
        x=[-female_top.get(c, 0) for c in common_causes],
        name="Female",
        orientation="h",
        marker_color=["red" if c == "Ischaemic heart disease" else "rgba(255,0,0,0.4)" for c in common_causes]
    ))
    fig.add_trace(go.Bar(
        y=common_causes,
        x=[male_top.get(c, 0) for c in common_causes],
        name="Male",
        orientation="h",
        marker_color=["black" if c == "Ischaemic heart disease" else "rgba(0,0,0,0.4)" for c in common_causes]
    ))

    tick_range = int(((max(female_top.max(), male_top.max()) // 100) + 1) * 100)
    fig.update_layout(
        barmode="relative",
        bargap=0.3,
        xaxis=dict(
            tickvals=[-tick_range, -tick_range//2, 0, tick_range//2, tick_range],
            ticktext=[str(abs(v)) for v in [-tick_range, -tick_range//2, 0, tick_range//2, tick_range]]
        ),
        template="simple_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
            margin=dict(l=10, r=10, t=10, b=10),
            width=900,     
            height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h5 style='margin-bottom: 10px;'>Global Distribution of Heart Disease Burden (DALYs per 100,000)</h5>", unsafe_allow_html=True)
    map_df = filtered_df.groupby("Location", as_index=False)["Value"].sum()
    map_fig = px.choropleth(
        map_df,
        locations="Location",
        locationmode="country names",
        color="Value",
        color_continuous_scale="Reds",
        height=400
    )
    map_fig.update_layout(
        coloraxis_colorbar=dict(
            title="DALYs per 100k",
            x=0.01,
            y=0.95,
            xanchor='left',
            yanchor='top',
            lenmode="pixels",
            len=200,
            thickness=12
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        width=1000,
        height=250,
    )
    st.plotly_chart(map_fig, use_container_width=True)

# --- COLUMN 3 ---
with col3:
    cardio_data = cardio_data.copy()
    cardio_data_filtered = cardio_data[(cardio_data["age_years"] >= age_min) & (cardio_data["age_years"] <= age_max)]
    if gender != "Both":
        cardio_data_filtered = cardio_data_filtered[cardio_data_filtered["gender_label"] == gender]
    
    # Replace cholesterol categories with estimated LDL levels in mg/dL
    cardio_data_filtered["cholesterol"] = cardio_data_filtered["cholesterol"].map({
        1: 95,    # Normal LDL
        2: 145,   # Above normal LDL
        3: 180    # High LDL
    })

    # Replace glucose categories with estimated fasting glucose in mg/dL
    cardio_data_filtered["gluc"] = cardio_data_filtered["gluc"].map({
        1: 90,    # Normal glucose
        2: 110,   # Prediabetic glucose
        3: 140    # Diabetic glucose
    })

    feature_map = {
            "BMI": "Body Mass Index",
            "ap_hi": "Systolic<br>Blood Pressure",
            "ap_lo": "Diastolic<br>Blood Pressure",
            "cholesterol": "LDL<br>Cholesterol",
            "gluc": "Fasting<br>Glucose",
            "smoke": "Smoking",
            "alco": "Alcohol<br>Intake",
            "active": "Physical<br>Activity"
        }
    radar_features = list(feature_map.keys())
    avg_profiles = cardio_data_filtered.groupby("cardio")[radar_features].mean().reset_index()
    avg_profiles["BMI"] *=  4 #BMI for better visualization
    std_profiles = avg_profiles.rename(columns=feature_map)

    clinical = [feature_map[k] for k in ["BMI", "ap_hi", "ap_lo", "cholesterol", "gluc"]]
    behavioral = [feature_map[k] for k in ["smoke", "alco", "active"]]

    st.markdown("<h5 style='margin-bottom: 5px;'>Differences in Clinical and Behavioral Profiles by Heart Disease Status</h5>", unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "polar"}, {"type": "polar"}]], subplot_titles=["Clinical", "Behavioral"])

    fig.add_trace(go.Scatterpolar(r=std_profiles[std_profiles["cardio"] == 0][clinical].values.flatten(), theta=clinical, fill='toself', name='No Heart Disease', line_color='black', opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=std_profiles[std_profiles["cardio"] == 1][clinical].values.flatten(), theta=clinical, fill='toself', name='Heart Disease', line_color='red', opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=std_profiles[std_profiles["cardio"] == 0][behavioral].values.flatten(), theta=behavioral, fill='toself', name='No Heart Disease', line_color='black', opacity=0.5, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatterpolar(r=std_profiles[std_profiles["cardio"] == 1][behavioral].values.flatten(), theta=behavioral, fill='toself', name='Heart Disease', line_color='red', opacity=0.5, showlegend=False), row=1, col=2)
    
    fig.update_layout(
    polar=dict(
        domain=dict(x=[0.15, 0.45]),  # Moves first plot more right
        radialaxis=dict(showticklabels=False)
        ),
        polar2=dict(
            domain=dict(x=[0.65, 0.95]),  # Moves second plot more right
            radialaxis=dict(showticklabels=False)
        ),
        height=180,
        margin=dict(t=20, b=20),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h5 style='margin-bottom: 5px;'>Factors Most Strongly Associated with Heart Disease</h5>", unsafe_allow_html=True)
    
    feature_map = {
            "BMI": "Body Mass Index",
            "ap_hi": "Systolic Blood Pressure",
            "ap_lo": "Diastolic Blood Pressure",
            "cholesterol": "LDL Cholesterol (mg/dL)",
            "gluc": "Fasting Glucose (mg/dL)",
            "smoke": "Smoking",
            "alco": "Alcohol Intake",
            "active": "Physical Activity"
        }
    corr = cardio_data_filtered[radar_features + ["cardio"]].corr()["cardio"].drop("cardio").sort_values()
    corr.index = corr.index.map(feature_map)

    fig, ax = plt.subplots(figsize=(4, 1.4))
    sns.barplot(x=corr.values, y=corr.index, palette=sns.color_palette("RdGy", len(corr)), ax=ax)
    ax.set_xlabel("Correlation with Heart Disease")
    ax.set_ylabel("")
    st.pyplot(fig, clear_figure=True)

    # Heatmap of disease prevalence by age/gender
    manual_data = {
        "Female": [0.0, 0.0, 0.0, 0.0, 3.0, 6.8, 9.9, 18.7],
        "Male":   [0.0, 0.0, 0.0, 0.0, 4.3, 11.4, 19.5, 31.3]
    }
    heatmap_data = pd.DataFrame(manual_data, index=["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+", "75+"])

    # Show heatmap
    st.markdown("<h5 style='margin-bottom: 5px;'>How Heart Disease Risk Changes with Age and Gender</h5>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 1.4))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap=LinearSegmentedColormap.from_list("white_red", ["white", "red"]),
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Prevalence (%)"}
    )
    ax.set_xlabel("Gender")
    ax.set_ylabel("Age Group")
    st.pyplot(fig, clear_figure=True)
