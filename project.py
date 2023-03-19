import pandas as pd
import numpy as np
import h3
import streamlit as st
from catboost import CatBoostClassifier

def main():
  model = CatBoostClassifier()
  model.load_model("model")

  dist_model = CatBoostClassifier()
  dist_model.load_model('dist_model')

  start_h3 = st.text_input("Start H3", key="start_h3")

  end_h3 = st.text_input("End H3", key="end_h3")

  valhalla_time = st.text_input("Valhalla time", key="valhalla_time")

  distance_in_meters = st.text_input("Distance in meters", key="distance_in_meters")

  start_lat = h3.cell_to_latlng(start_h3)[0]
  start_lng = h3.cell_to_latlng(start_h3)[1]
  end_lat = h3.cell_to_latlng(end_h3)[0]
  end_lng = h3.cell_to_latlng(end_h3)[1]
  
  if st.button('Train model'):
    
    st.write(start_lat, start_lng, end_lat, end_lng, distance_in_meters, 0, valhalla_time)

    input_ = [start_lat, start_lng, end_lat, end_lng, distance_in_meters, 0, valhalla_time]
    model.predict(np.array(input_))
    st.write('Predicted time for Monday, 8 A.M.')

    input_ = [start_lat, start_lng, end_lat, end_lng, distance_in_meters, 1, valhalla_time]
    model.predict(np.array(input_))
    st.write('Predicted time for Thursday, 11 P.M.')

    input_ = [start_lat, start_lng, end_lat, end_lng, distance_in_meters, 2, valhalla_time]
    model.predict(np.array(input_))
    st.write('Predicted time for Sunday, 3 P.M.')

if __name__ == "__main__":
    main()
