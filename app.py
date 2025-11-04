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
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr,0)
    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)
    st.image(image)
    st.write('Wound in image is {} with an accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))

    predicted_label = data_cat[np.argmax(score)]
    if predicted_label == str('Mild diabetes'):
        st.write('Recommendations:')
        st.write('/tClean the wound starile saline and cover it with a sterile pad.')
        st.write('/tReduce pressure or wheight-bearing on the affected foot.')
        st.write('/tConsult a doctor if possible.')
        st.write('Things to check daily.')
        st.write('/tIt becomes red, swollen or more painful.')
    elif predicted_label == str('Moderate diabetes'):
        st.write('Recommendations:')
        st.write('/tVisit the hospital for wound care.')
        st.write('/tDO NOT treat the wound yourself.')
        st.write('/tClosely monitor blood sugar levels.')
        st.write('Things to check daily.')
        st.write('/tIt is darkening, swelling or abnormal changes.')
    elif predicted_label == str('Not a diabetic wound'):
        st.write('Do not worry. You do not have diabetes.')
        st.write("Cover it up with alchohol and don't let water get to it")
    else:
        st.write('Recommendations:')
        st.write('/tSeek immediate hospital treatment.')
        st.write('/tMaintain strict wound hegiene.')
        st.write('/tMoniter the wound daily under medical supervision AT ALL TIMES.')



