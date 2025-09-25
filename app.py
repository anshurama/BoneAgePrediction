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

def get_ConvNeXtXLarge_model():
    base_model = keras.applications.ConvNeXtXLarge(include_top=False, weights="imagenet", input_shape=(224, 224, 3), classes=1000, classifier_activation="softmax")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)
    return model_frozen


def load_sklearn_models(model_path):
    with open(model_path, "rb") as model_file:
        final_model=pickle.load(model_file)
    return final_model

ConvNeXtXLarge_featurize_model=get_ConvNeXtXLarge_model()
regression_model= load_sklearn_models("Best_MLP_Regressor")

def featurization(image_path,model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions=model.predict(img_preprocessed)
    return predictions



def run_app():
    st.title("Bone Age Prediction")
    st.subheader("Predict Bone Age")
    st.image(IMAGE_ADDRESS,caption="Bone Age")
    st.subheader("Please upload your image")
    gender = st.radio("select gender", ("Male", "Female"))
    gender_feature = 1 if gender == "Male" else 0
    image= st.file_uploader("Please upload your image", type = ["png", "jpg", "jpeg"], accept_multiple_files= False, help = "Please upload an image")
    if image:
        user_image = Image.open(image)
        user_image.save(IMAGE_NAME)    
        st.image(user_image, caption="User uploaded image")
        with st.spinner("processing..."):
            image_features=featurization(IMAGE_NAME, ConvNeXtXLarge_featurize_model)
            image_features_with_gender=np.append(image_features,[[gender_feature]], axis=1)
            model_predict=regression_model.predict(image_features_with_gender)
            result=model_predict[0]
            st.success(f"prediction:{result}")
run_app()