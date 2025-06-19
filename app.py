import streamlit as st
import pickle
import numpy as np
import pandas as pd
import data_preprocessing
from data_preprocessing import data_preprocessing, encoder_Marital_status, encoder_Application_mode, encoder_Course, encoder_Daytime_evening_attendance, encoder_Previous_qualification, encoder_Mothers_qualification, encoder_Fathers_qualification, encoder_Fathers_occupation, encoder_Mothers_occupation, encoder_Displaced, encoder_Debtor, encoder_Tuition_fees_up_to_date, encoder_Gender, encoder_Scholarship_holder

from prediction import prediction


st.title("Student Dropout Prediction")


# Kelompok Kurikulum (Curricular Group)
with st.form(key='curricular_form'):
    st.header("Kelompok Kurikulum")

    # Curricular Group Inputs (grouped in columns for layout)
    col1, col2, col3 = st.columns(3)
    with col1:
        admission_grade = st.number_input('Admission Grade', min_value=0, max_value=200, value=80)
    with col2:
        age_at_enrollment = st.number_input('Age at Enrollment', min_value=17, max_value=34, value=18)
    with col3:
        previous_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0, max_value=200, value=75)

    col1, col2 = st.columns(2)
    with col1:
        curricular_units_1st_sem_credited = st.number_input('Curricular Units 1st Sem Credited', min_value=0, max_value=20, value=12)
        curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Sem Enrolled', min_value=0, max_value=10, value=6)
        curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Sem Evaluations', min_value=0, max_value=20, value=10)
    with col2:
        curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem Approved', min_value=0, max_value=10, value=8)
        curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Sem Grade', min_value=0, max_value=20, value=7)
        curricular_units_1st_sem_without_evaluations = st.number_input('Curricular Units 1st Sem Without Evaluations', min_value=0, max_value=10, value=0)

    col1, col2 = st.columns(2)
    with col1:
        curricular_units_2nd_sem_credited = st.number_input('Curricular Units 2nd Sem Credited', min_value=0, max_value=20, value=12)
        curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Sem Enrolled', min_value=0, max_value=10, value=6)
        curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Sem Evaluations', min_value=0, max_value=20, value=8)
    with col2:
        curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem Approved', min_value=0, max_value=10, value=6)
        curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Sem Grade', min_value=0, max_value=20, value=7)
        curricular_units_2nd_sem_without_evaluations = st.number_input('Curricular Units 2nd Sem Without Evaluations', min_value=0, max_value=10, value=0)

    st.form_submit_button("Submit")

# Kelompok Informasi Pribadi (Personal Information Group)
with st.form(key='personal_info_form'):
    st.header("Kelompok Informasi Pribadi")

    col1, col2, col3 = st.columns(3)
    with col1:
        marital_status = st.selectbox('Marital Status', encoder_Marital_status.classes_, index=0)
        application_mode = st.selectbox('Application Mode', encoder_Application_mode.classes_, index=0)
    with col2:
        course = st.selectbox('Course', encoder_Course.classes_, index=0)
        daytime_evening_attendance = st.selectbox('Daytime/Evening Attendance', encoder_Daytime_evening_attendance.classes_, index=0)
    with col3:
        previous_qualification = st.selectbox('Previous Qualification', encoder_Previous_qualification.classes_, index=0)
        mothers_qualification = st.selectbox('Mother\'s Qualification', encoder_Mothers_qualification.classes_, index=0)

    col1, col2 = st.columns(2)
    with col1:
        fathers_qualification = st.selectbox('Father\'s Qualification', encoder_Fathers_qualification.classes_, index=0)
        fathers_occupation = st.selectbox('Father\'s Occupation', encoder_Fathers_occupation.classes_, index=0)
    with col2:
        mothers_occupation = st.selectbox('Mother\'s Occupation', encoder_Mothers_occupation.classes_, index=0)
        displaced = st.selectbox('Displaced (Yes/No)', encoder_Displaced.classes_, index=0)

    col1, col2 = st.columns(2)
    with col1:
        debtor = st.selectbox('Debtor (Yes/No)', encoder_Debtor.classes_, index=0)
        tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date (Yes/No)', encoder_Tuition_fees_up_to_date.classes_, index=0)
    with col2:
        gender = st.selectbox('Gender', encoder_Gender.classes_, index=0)
        scholarship_holder = st.selectbox('Scholarship Holder (Yes/No)', encoder_Scholarship_holder.classes_, index=0)

    st.form_submit_button("Submit")

# Prepare Data for Prediction
data = {
    "Admission_grade": [admission_grade],
    "Age_at_enrollment": [age_at_enrollment],
    "Previous_qualification_grade": [previous_qualification_grade],
    "Curricular_units_1st_sem_credited": [curricular_units_1st_sem_credited],
    "Curricular_units_1st_sem_enrolled": [curricular_units_1st_sem_enrolled],
    "Curricular_units_1st_sem_evaluations": [curricular_units_1st_sem_evaluations],
    "Curricular_units_1st_sem_approved": [curricular_units_1st_sem_approved],
    "Curricular_units_1st_sem_grade": [curricular_units_1st_sem_grade],
    "Curricular_units_1st_sem_without_evaluations": [curricular_units_1st_sem_without_evaluations],
    "Curricular_units_2nd_sem_credited": [curricular_units_2nd_sem_credited],
    "Curricular_units_2nd_sem_enrolled": [curricular_units_2nd_sem_enrolled],
    "Curricular_units_2nd_sem_evaluations": [curricular_units_2nd_sem_evaluations],
    "Curricular_units_2nd_sem_approved": [curricular_units_2nd_sem_approved],
    "Curricular_units_2nd_sem_grade": [curricular_units_2nd_sem_grade],
    "Curricular_units_2nd_sem_without_evaluations": [curricular_units_2nd_sem_without_evaluations],
    "Marital_status": [marital_status],
    "Application_mode": [application_mode],
    "Course": [course],
    "Daytime_evening_attendance": [daytime_evening_attendance],
    "Previous_qualification": [previous_qualification],
    "Mothers_qualification": [mothers_qualification],
    "Fathers_qualification": [fathers_qualification],
    "Fathers_occupation": [fathers_occupation],
    "Mothers_occupation": [mothers_occupation],
    "Displaced": [displaced],
    "Debtor": [debtor],
    "Tuition_fees_up_to_date": [tuition_fees_up_to_date],
    "Gender": [gender],
    "Scholarship_holder": [scholarship_holder]
}

# Convert ke DataFrame sebelum diproses
df_data = pd.DataFrame(data)

with st.expander("View the Raw Data"):
    st.dataframe(data=df_data, width=800, height=80)

if st.button('Predict'):
    new_data = data_preprocessing(data=df_data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=80)
    st.write("Result: {}".format(prediction(new_data)))