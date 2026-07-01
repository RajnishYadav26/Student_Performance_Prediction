import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Student Performance Prediction")

st.write("Enter student details below.")

model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
gender_options = {
    "Female": 0,
    "Male": 1
}

ethnicity_options = {
    "Group A": 0,
    "Group B": 1,
    "Group C": 2,
    "Group D": 3
}

parent_edu_options = {
    "Level 1": 1,
    "Level 2": 2,
    "Level 3": 3,
    "Level 4": 4,
    "Level 5": 5
}

tutoring_options = {
    "No": 0,
    "Yes": 1
}

support_options = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Very High": 3
}

extra_options = {
    "No": 0,
    "Yes": 1
}

sports_options = {
    "No": 0,
    "Yes": 1
}

music_options = {
    "No": 0,
    "Yes": 1
}

volunteer_options = {
    "No": 0,
    "Yes": 1
}

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age", 15, 25, 17)

    gender = st.selectbox(
        "Gender",
        list(gender_options.keys())
    )
    gender = gender_options[gender]

    ethnicity = st.selectbox(
        "Ethnicity",
        list(ethnicity_options.keys())
    )
    ethnicity = ethnicity_options[ethnicity]

    parent_edu = st.selectbox(
        "Parental Education",
        list(parent_edu_options.keys())
    )
    parent_edu = parent_edu_options[parent_edu]

    study = st.slider(
        "Study Time Weekly",
        0.0,
        25.0,
        10.0
    )

with col2:

    absence = st.slider(
        "Absences",
        0,
        30,
        5
    )

    tutoring = st.selectbox(
        "Tutoring",
        list(tutoring_options.keys())
    )
    tutoring = tutoring_options[tutoring]

    support = st.selectbox(
        "Parental Support",
        list(support_options.keys())
    )
    support = support_options[support]

    extra = st.selectbox(
        "Extracurricular Activities",
        list(extra_options.keys())
    )
    extra = extra_options[extra]

    sports = st.selectbox(
        "Sports",
        list(sports_options.keys())
    )
    sports = sports_options[sports]

    music = st.selectbox(
        "Music",
        list(music_options.keys())
    )
    music = music_options[music]

    volunteer = st.selectbox(
        "Volunteering",
        list(volunteer_options.keys())
    )
    volunteer = volunteer_options[volunteer]

    gpa = st.slider(
        "GPA",
        0.0,
        4.0,
        3.0
    )

if st.button("Predict"):

    data = np.array([[
        age,
        gender,
        ethnicity,
        parent_edu,
        study,
        absence,
        tutoring,
        support,
        extra,
        sports,
        music,
        volunteer,
        gpa
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]
    confidence = max(probabilities) * 100

    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    st.markdown("---")
    st.subheader("Prediction Result")
    st.metric("Predicted Grade Class", int(prediction))

    if prediction == 0:
        st.success("Excellent Performance")
    elif prediction == 1:
        st.success("Good Performance")
    elif prediction == 2:
        st.warning("Average Performance")
    else:
        st.error("Needs Improvement")