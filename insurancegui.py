import streamlit as st
import pickle 
import numpy as np


pickle_in = open('insurance1.pkl','rb')
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

st.markdown('<p class="big-font">Insurance price prediction</p>', unsafe_allow_html=True)


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