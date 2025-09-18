import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pickle
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
IMAGE_ADDRESS = "https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-022-05282-z/MediaObjects/41598_2022_5282_Fig1_HTML.png"
IMAGE_SIZE = (224,224)
IMAGE_NAME = "User_image.png"
class_label = ["Male", "Female"]
class_label.sort()

def run_app():
    st.title("Bone Age Prediction")
    st.subheader("Predict Bone Age")
    st.image(IMAGE_ADDRESS,caption="Bone Age")
    st.subheader("Please upload your image")
    gender = st.radio("select gender", ("Male", "Female"))
    gender_feature = 1 if gender == "Male" else 0
    image= st.file_uploader("Please upload your image", type = ["png", "jpg", "jpeg"], accept_multiple_files= False, help = "Please upload an image")
                            
run_app()