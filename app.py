import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load and preprocess the dataset
df = pd.read_excel("xgb3_preprocess1.xlsx")

# Create a 'Fee_category' column
def category(x):
    if 0 <= x < 100:
        return '0-100'
    elif 100 <= x < 250:
        return '100-250'
    elif 250 <= x < 400:
        return '250-400'
    elif 400 <= x < 600:
        return '400-600'
    elif 600 <= x < 800:
        return '600-800'
    else:
        return '800-950'

df['Fee_category'] = df['Fees'].apply(category)

# Handle skewness in 'Experience'
df['Experience'] = np.sqrt(df['Experience'])

# Encoding categorical variables using LabelEncoder
label_encoders = {}
categorical_cols = ['Place', 'Profile', 'Fee_category']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scaling
scaler = MinMaxScaler()
cols_to_scale = df.columns.difference(['Fees'])
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Split data into features and target
x = df.copy()
x.drop('Fees', axis=1, inplace=True)
y = df['Fees']

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)

# Train the model
finalmodel = XGBRegressor()
finalmodel.fit(xtrain, ytrain)

# Streamlit app
st.title("Doctor Fee Prediction App")

# User inputs
experience = st.number_input("Years of Experience", min_value=0, max_value=66, value=0, step=1)
rating = st.number_input("Doctor Rating", min_value=1, max_value=100, value=50, step=1)
place = st.selectbox("Place", label_encoders['Place'].classes_)
profile = st.selectbox("Doctor Specialization", label_encoders['Profile'].classes_)
miscellaneous_info = st.selectbox("Miscellaneous Info Existent", ["Not Present", "Present"])
num_of_qualifications = st.number_input("Number of Qualifications", min_value=1, max_value=10, value=1, step=1)

# Prediction button
if st.button("Predict"):
    # Encoding user input using LabelEncoder
    place_encoded = label_encoders['Place'].transform([place])[0]
    profile_encoded = label_encoders['Profile'].transform([profile])[0]
    misc_info_encoded = 1 if miscellaneous_info == 'Present' else 0

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Experience': [np.sqrt(experience)],
        'Rating': [rating],
        'Place': [place_encoded],
        'Profile': [profile_encoded],
        'Miscellaneous_Info': [misc_info_encoded],
        'Num_of_Qualifications': [num_of_qualifications],
        'Fee_category': [0]  # Fee category is set to 0
    })

    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Prediction
    prediction = finalmodel.predict(input_data)

    # Output the prediction
    st.write(f"Predicted Doctor Fee: {np.round(prediction[0], 2)}")
