import streamlit as st
import pickle 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("insurance.csv")

encoder = LabelEncoder()
encoded_gender = encoder.fit_transform(data['sex'])
data_sex = pd.DataFrame(encoded_gender)
data['sex_encoded'] = data_sex
encoded_smoker = encoder.fit_transform(data['smoker'])
data['encoded_smoker'] = encoded_smoker


data.drop(['sex','smoker'], axis=1, inplace=False)
data = data.loc[:,['age','bmi','children','region','sex_encoded','encoded_smoker','charges']]

scaler =  MinMaxScaler()
scale_data =scaler.fit_transform(data[['age','bmi']])
data12 = pd.DataFrame(scale_data)
X = data.drop(['charges','region'],axis=1)
y= data['charges'] # target data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)


with open('insurance.pkl','wb') as file:
    pickle.dump(model,file)
    file.close()




pickle_in = open('insurance.pkl','rb')
insurance = pickle.load(pickle_in)


def insurance_predict(age, bmi,no_of_children,gender_index,smoker_index):
    input = np.array([age, bmi,no_of_children,gender_index,smoker_index]).reshape(1,-1)
    prediction = insurance.predict(input)
    return prediction

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:60px ;
    color :#05B6F7 ;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Insurance price predictions</p>', unsafe_allow_html=True)


age= st.text_input("Enter your age ")
bmi = st.text_input("Enter Bmi ")
no_of_children = st.text_input("Enter number of children")

gender_option =['female','male']
gender =st.radio("Gender", gender_option)  
gender_index =  gender_option.index(gender)

smoker_option =['No','Yes']
smoker =st.radio("Are you smoker", smoker_option)  
smoker_index =  smoker_option.index(smoker)
result=""
if st.button("predict"):
    result = insurance_predict(int(age),float(bmi),int(no_of_children),int(gender_index),int(smoker_index))
st.success(result)
