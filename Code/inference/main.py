from __future__ import print_function
import argparse
from PIL import Image
import io
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import os
import pickle
from pickle import dump, load
import pandas as pd
import nltk
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import concatenate
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def word_for_id(integer, tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def load_resources():
    global model, tokenizer


    model = load_model("/data/model_8k_29.h5")
    with open("/data/tokenizer.p", "rb") as f:
        tokenizer = pickle.load(f)

def get_inference(image):
    image = Image.fromarray(image)
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    xception_model = Xception(include_top=False, pooling="avg")
    feature = xception_model.predict(image)
    description = generate_desc(model, tokenizer, feature, 32)
    return description

demo = gr.Interface(fn=get_inference,
                     inputs=gr.Image(),
                     outputs="text",
                     title="Image Caption Generator",
                     description="Upload an image and let the trained \
                      CNN-LSTM model generate a caption describing it.")

if __name__ == "__main__":
    load_resources()
    demo.launch(share=True)
