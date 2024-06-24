import base64
from io import BytesIO

import firebase_admin
import gradio as gr
import numpy as np
from fastai.vision.all import *
import json
from firebase_admin import credentials, firestore
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

cred = credentials.Certificate("firebase_key.json")
app = firebase_admin.initialize_app(cred)
db = firestore.client()

learn = load_learner("model.pkl")
names = json.load(open("./translations.json"))


def classify(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    pred, idx, probs = learn.predict(np.asarray(image))

    db.collection("preds").add(  # inilo db
        {
            "image": img_str,
            "prediction": pred.title(),
            "time_added": firestore.SERVER_TIMESTAMP,
        }
    )

    return [
        names[pred]["id"],
        names[pred]["en"],
        f"./audios/en/{pred}.mp3",
        f"./audios/id/{pred}.mp3",
    ]


with gr.Blocks(
    css=".gradio-container {background-image: url('file=Fruitzone.jpg')}"
) as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Webcam(label="Gambar", shape=(200, 200), type="pil")
            predict_btn = gr.Button("Cek Nama Buah Yuk", variant="primary")

        with gr.Column():
            id_fruit_name = gr.Label(label="Bahasa indonesia")
            en_fruit_name = gr.Label(label="Bahasa inggris")
            en_audio = gr.Audio(label="Audio inggris")
            id_audio = gr.Audio(label="Audio indonesia")

    predict_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[id_fruit_name, en_fruit_name, id_audio, en_audio],
        api_name="classify_image",
    )

demo.launch()
