import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Only do this if 'model.pkl' already exists
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# If you want to create/save the model, do that in a separate script
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

st.header('Car Price ML Model')

df = pd.read_csv("C:\\Users\\ofurh\\Downloads\\Cardetails.csv")

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

df['name'] = df['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', df['name'].unique())
year = st.slider('Select Year', 1994, 2024)
km_driven = st.slider('No of kms driven', 0, 100000)
fuel = st.selectbox('Fuel Type', df['fuel'].unique())
seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
transmission = st.selectbox('Transmission Type', df['transmission'].unique())
owner = st.selectbox('Owner Type', df['owner'].unique())
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 50, 500)
seats = st.slider('Seats', 4, 10)


if st.button('Predict Price'):
    input_data_model = pd.DataFrame(
        [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']
    )
  
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                           [1,2,3,4,5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)
    st.write(input_data_model)

    car_price = model.predict(input_data_model)
    st.markdown(f'Car Price is going to be **₹{car_price[0]:,.2f}**')


