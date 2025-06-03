import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import streamlit as st

#load the trained model
model=tf.keras.models.load_model('modeal.h5')

#load the encoders and scaler

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('One_hot_encoder.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('Scaler.pkl','rb') as file:
    scaler=pickle.load(file)

#streamlit app

st.title('CUstomer Churn Prediction')

#user Input

geography= st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age= st.slider('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit score')
est_sal = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_prod= st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Jas Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member ?',[0,1])

#input Data

input_data = pd.DataFrame({
    'Credit_score':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'Num_of_prod':[num_of_prod],
    'HasCrCard':[has_cr_card],
    'Is_Active_mem':[is_active_member],
    'Estimated_salary':[est_sal]
})

#one Hot Encode "Geography"
geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combine One Hot Encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input

input_data_scaled = scaler.fit_transform(input_data)

#Predict churn

pred=model.predict(input_data_scaled)
pred_prob = pred[0][0]

st.write("the churn probability is: ",pred_prob)

if pred_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
