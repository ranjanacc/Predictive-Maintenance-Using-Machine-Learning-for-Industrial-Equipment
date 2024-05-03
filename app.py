import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
st.title('PdM Web App') 
st.markdown('To predict machine health instantly.')
air = st.number_input('Air Temperature [K]:') 
process = st.number_input('Process Temperature [K]:') 
rotation = st.number_input('Rotational Speed[rpm]:') 
torque = st.number_input('Torque [Nm]:') 
tool_wear = st.number_input('Tool Wear [min]:') 
result = 10
if st.button("Predict"):
	result = predict(np.array([[air, process, rotation, torque, tool_wear]]))
if result == 0:
	st.text("Machine is in optimal state.")
elif result == 1:
	st.text("Machine is about to fail. Take preemptive care now.")

