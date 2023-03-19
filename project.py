import pandas as pd
import numpy as np
import h3
import streamlit as st
from catboost import CatBoostRegressor

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def main():
  model = CatBoostRegressor()
  model.load_model("model")

  dist_model = CatBoostRegressor()
  dist_model.load_model('dist_model')

  start_h3 = st.text_input("Start H3", key="start_h3")

  end_h3 = st.text_input("End H3", key="end_h3")
  
  distance_in_meters = st.text_input("Distance in meters", key="distance_in_meters")
  distance_in_meters = int(distance_in_meters)
  
  valhalla_time = st.text_input("Valhalla time", key="valhalla_time")
  valhalla_time = int(valhalla_time)

  start_lat = h3.cell_to_latlng(start_h3)[0]
  start_lng = h3.cell_to_latlng(start_h3)[1]
  end_lat = h3.cell_to_latlng(end_h3)[0]
  end_lng = h3.cell_to_latlng(end_h3)[1]
  
  if st.button('Predict time'):
    st.write('Initial coordinates:', h3.cell_to_latlng(start_h3))
    st.write('Final coordinates:', h3.cell_to_latlng(end_h3))
             
    input_ = [start_lat, start_lng, end_lat, end_lng, distance_in_meters, 1, valhalla_time]
    st.write('Predicted time for Monday, 8 A.M.', convert(model.predict(np.array(input_))))

    input_ = [start_lat, start_lng, end_lat, end_lng, distance_in_meters, 2, valhalla_time]
    st.write('Predicted time for Thursday, 11 P.M.', convert(model.predict(np.array(input_))))

    input_ = [start_lat, start_lng, end_lat, end_lng, distance_in_meters, 3, valhalla_time]    
    st.write('Predicted time for Sunday, 3 P.M.', convert(model.predict(np.array(input_))))

if __name__ == "__main__":
    main()
