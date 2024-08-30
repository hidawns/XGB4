import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_excel("xgb3_preprocess1.xlsx")

# Preprocessing functions
def category(x):
    if x >= 0 and x < 100:
        return '0-100'
    elif x >= 100 and x < 250:
        return '100-250'
    elif x >= 250 and x < 400:
        return '250-400'
    elif x >= 400 and x < 600:
        return '400-600'
    elif x >= 600 and x < 800:
        return '600-800'
    else:
        return '800-950'

df['Fee_category'] = df['Fees'].apply(category)
df['Experience'] = np.sqrt(df['Experience'])

encoder = LabelEncoder()
cols_to_encode = ['Place', 'Profile', 'Fee_category']
for col in cols_to_encode:
    df[col] = encoder.fit_transform(df[col])

scaler = MinMaxScaler()
cols_to_scale = df.columns.difference(['Fees'])
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

x = df.copy()
x.drop('Fees', axis=1, inplace=True)
y = df['Fees']

# Split the data for model training
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)

# Train the final model
finalmodel = XGBRegressor()
finalmodel.fit(xtrain, ytrain)

# Streamlit App
st.title("Doctor Consultation Fee Prediction")

# User input for the features
experience = st.number_input("Years of Experience", min_value=0, max_value=66, step=1)
num_of_qualifications = st.number_input("Number of Qualifications", min_value=1, max_value=10, step=1)
rating = st.number_input("Doctor Rating", min_value=1, max_value=100, step=1)
miscellaneous_info = st.selectbox("Miscellaneous Info Existent", ['Not Present', 'Present'])
place = st.selectbox("Place", ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Coimbatore', 'Ernakulam', 'Thiruvananthapuram', 'Other'])
profile = st.selectbox("Doctor Specialization", ['Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 'General Medicine', 'Homeopath'])

# Encoding user input
place_encoded = encoder.transform([place])[0]
profile_encoded = encoder.transform([profile])[0]
misc_info_encoded = 1 if miscellaneous_info == 'Present' else 0

# Creating a dataframe from user input
input_data = pd.DataFrame({
    'Experience': [np.sqrt(experience)],
    'Rating': [rating],
    'Place': [place_encoded],
    'Profile': [profile_encoded],
    'Miscellaneous_Info': [misc_info_encoded],
    'Num_of_Qualifications': [num_of_qualifications],
    'Fee_category': [0.0]  # Set Fee_category to 0
})

# Scaling user input
input_data[cols_to_scale] = scaler.transform(input_data)

# Predicting the fee
prediction = finalmodel.predict(input_data)

# Display the result
st.write(f"The predicted consultation fee is: ₹{prediction[0]:.2f}")
