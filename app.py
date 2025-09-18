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

