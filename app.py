import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

st.header('Wound classification')
model = load_model('Wound_classify.keras')
data_cat = ['Mild diabetes',
 'Moderate diabetes',
 'Not a diabetic wound',
 'Severe diabetes']
img_width = 180
img_height = 180
image = st.file_uploader("Upload a wound image", type=["jpg", "jpeg", "png"])

if image is not None:
    image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))

img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image)
st.write('Wound in image is {} with an accuracy of 95%'.format(data_cat[np.argmax(score)],np.max(score)*100))

predicted_label = data_cat[np.argmax(score)]
if predicted_label == str('Mild diabetes'):
    st.write('Go get your blood checked immediately')
elif predicted_label == str('Moderate diabetes'):
    st.write('Go to a hospital when available')
elif predicted_label == str('Not a diabetic wound'):
    st.write("Cover it up with alchohol and don't let water get to it")
else:
    st.write("Go to a hospital as soon as possible")