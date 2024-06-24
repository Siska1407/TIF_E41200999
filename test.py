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

with gr.Blocks(
    css=".gradio-container {background-image: url('file=Background/4.jpeg');background-size: cover; background-size: 100% 100%;}"  # .block.svelte-kz0ejz{background-color: rgba(0,0,0,0);} kalo butuh transparan
) as demo:
    gr.Markdown(" ")
    gr.Markdown(" ")
    with gr.Row():
        with gr.Column():
            image_input = gr.Webcam(label="Gambar", shape=(200, 200), type="pil")
            with gr.Row():
                predict_id_btn = gr.Button("Bahasa Indonesia", variant="primary")
                predict_en_btn = gr.Button("Bahasa Inggris", variant="secondary")

        with gr.Column():
            with gr.Column(visible=False) as id_fruit_name_col:
                id_fruit_name = gr.Label(label="Bahasa indonesia")
                id_audio = gr.Audio(label="Audio indonesia")
            with gr.Column(visible=False) as en_fruit_name_col:
                en_fruit_name = gr.Label(label="Bahasa inggris")
                en_audio = gr.Audio(label="Audio inggris")

    def classify_id(image):
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
        id_fruit_name_col = gr.update(visible=True)
        en_fruit_name_col = gr.update(visible=False)
        return [
            names[pred]["id"],
            f"./audios/id/" + names[pred]["id"] + ".mp3",
            id_fruit_name_col,
            en_fruit_name_col,
        ]

    def classify_en(image):
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
        en_fruit_name_col = gr.update(visible=True)
        id_fruit_name_col = gr.update(visible=False)
        return [
            names[pred]["en"],
            f"./audios/en/" + names[pred]["id"] + ".mp3",
            id_fruit_name_col,
            en_fruit_name_col,
        ]

    predict_id_btn.click(
        fn=classify_id,
        inputs=image_input,
        outputs=[id_fruit_name, id_audio, id_fruit_name_col, en_fruit_name_col],
        api_name="classify_image",
    )
    predict_en_btn.click(
        fn=classify_en,
        inputs=image_input,
        outputs=[en_fruit_name, en_audio, id_fruit_name_col, en_fruit_name_col],
        api_name="classify_image",
    )


# demo.launch(share=True)
demo.launch()
