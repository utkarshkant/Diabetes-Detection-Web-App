'''
# Objective
- Build a machine learning model to detect diabetes
- Make a web app from that model
'''

# import libraries
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st 

# create a title and sub-title for the web-app
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning!
""")

# open and display an image
image = Image.open(r"F:\YouTube\# Computer Science_YT\# In Progress\Diabetes Web App\diabetes.jpg")
st.image(image, caption="Diabetes ML", use_column_width=True)

# load dataset
# Source: https://www.kaggle.com/saurabh00007/diabetescsv
df = pd.read_csv('diabetes.csv')

# set a subheader on the webapp
st.subheader("Data Information")

# display data as a table
st.dataframe(df)

# display data statistics
st.write(df.describe())

# display data as chart
chart = st.bar_chart(df)

# split data into independent 'X' and dependent 'y' features
X = np.array(df.drop(['Outcome'], axis=1))
y = np.array(df['Outcome'])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# get user inputs for different features
def getUserInputs():
    pregnancies = st.sidebar.slider('pregnancies', float(df['Pregnancies'].min()), float(df['Pregnancies'].max()), float(df['Pregnancies'].mean()))
    glucose = st.sidebar.slider('glucose', float(df['Glucose'].min()), float(df['Glucose'].max()), float(df['Glucose'].mean()))
    blood_pressure = st.sidebar.slider('blood_pressure', float(df['BloodPressure'].min()), float(df['BloodPressure'].max()), float(df['BloodPressure'].mean()))
    skin_thickness = st.sidebar.slider('skin_thickness', float(df['SkinThickness'].min()), float(df['SkinThickness'].max()), float(df['SkinThickness'].mean()))
    insulin = st.sidebar.slider('insulin', float(df['Insulin'].min()), float(df['Insulin'].max()), float(df['Insulin'].mean()))
    BMI = st.sidebar.slider('BMI', float(df['BMI'].min()), float(df['BMI'].max()), float(df['BMI'].mean()))
    DPF = st.sidebar.slider('DPF', float(df['DiabetesPedigreeFunction'].min()), float(df['DiabetesPedigreeFunction'].max()), float(df['DiabetesPedigreeFunction'].mean()))
    age = st.sidebar.slider('age', float(df['Age'].min()), float(df['Age'].max()), float(df['Age'].mean()))

    # store dictionary
    user_data = {
                "pregnancies":pregnancies,
                "glucose":glucose,
                "blood_pressure":blood_pressure,
                "skin_thickness":skin_thickness,
                "insulin":insulin,
                "BMI":BMI,
                "DPF":DPF,
                "age":age
                }

    # transform data into data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

# store users input
user_input = getUserInputs()

# set a subheader and display user's inputs
st.subheader("User Inputs:")
st.write(user_input)

# train model
rcf = RandomForestClassifier()
rcf.fit(X_train, y_train)

# model predictions
preds = rcf.predict(X_test)

# model evaluation
st.subheader("Model Test Accuracy Score:")
st.write(str(round(accuracy_score(y_test, preds)*100, 2)) + "%")

# prediction on user's inputs
user_preds = rcf.predict(user_input)

# set a subheader
st.subheader("Diabetes Detection")
st.write("0=No, 1=Yes")
st.write(user_preds)