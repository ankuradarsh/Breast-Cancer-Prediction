# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:55:18 2023

@author: Ankur Adarsh
"""

# -*- coding: utf-8 -*-
"""
@author: Ankur Adarsh
"""

import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_svm_svc_model = pickle.load(open("trained_svm_svc_model.sav", "rb"))

#creating the function for prediction
def prediction_function(data_to_predict):
    input_data = np.asarray(data_to_predict)
    data_to_predict_reshaped = input_data.reshape(1,-1)
    data_pred = loaded_svm_svc_model.predict(data_to_predict_reshaped)
    if(data_pred[0]==1):
        return "Prediction is Malignant (cancerous)"
    else:
        return "Prediction is Benign (non-cancerous)"
        

def main():
    st.title("Breast Cancer Prediction Web App")
    
    features = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    input_features = []
    for value in features:
        input_features=st.text_input(f"Input for {value}")
        
    result=""
    if st.button("Test Result"):
        result = prediction_function(input_features)
        
    st.success(result)
    
if __name__ == "__main__":
    main()
