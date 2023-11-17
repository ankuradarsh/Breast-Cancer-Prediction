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
loaded_svm_svc_model = pickle.load(open("trained_decision_tree_model.sav", "rb"))


#data_to_predict = (842302, 17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189)
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
    
    features =['ID', 'Radius Mean', 'Texture Mean', 'Perimeter Mean',
       'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean',
       'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
       'Radius se', 'Texture se', 'Perimeter se', 'Area se', 'Smoothness se',
       'Compactness se', 'Concavity se', 'Concave points se', 'Symmetry se',
       'Fractal Dimension se', 'Radius worst', 'Texture worst',
       'Perimeter worst', 'Area worst', 'Smoothness worst',
       'Compactness worst', 'Concavity worst', 'Concave Points worst',
       'Symmetry worst', 'Fractal Dimension worst']
    input_features = []
    for value in features:
        input_features.append(st.text_input(f"Input for {value}"))
        
    result=""
    if st.button("Test Result"):
        result = prediction_function(input_features)
        input_features.clear()
        
    st.success(result)
    
if __name__ == "__main__":
    main()
