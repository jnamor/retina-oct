import streamlit as st
import torch
from model import ConvNet, Model


st.write("""
         # Predict Retina Damage From Optical Coherence Tomography (OCT)
         """
         )
st.write("This is a simple image classification web app to predict retina damage")
file = st.file_uploader("Please upload an image file", type=["jpeg", "jpg", "png"], accept_multiple_files=True)

def predic_model(classes):
    checkpoint = torch.load('best_checkpoint.model')
    model = ConvNet(len(classes))
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def check_answers(qsdsq, d = None):
    if d is None:
        d = {}

    good_answer = 0
    for key, value in qsdsq.items():
        tested_element = key.split('-')[0]
        if tested_element == value:
            good_answer += 1
        else:
            d[key] = value

    predicted_value = (good_answer/len(qsdsq)) * 100

    return predicted_value, d

transformer = ConvNet.transformer(150, 150)
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
model = predic_model(classes)
PredictModel = Model(model, classes)

pred_dict = {}

for img in file:
    pred_dict[img.name] = PredictModel.prediction(img, transformer)

if pred_dict:
    predicted_value, wrong_answers = check_answers(pred_dict)
    st.write(f"Dict: {wrong_answers}")
    st.write(f"Predicted value: {predicted_value}%")



