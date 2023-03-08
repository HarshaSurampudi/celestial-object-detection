from fastai.vision.all import *
import gradio as gr

learn = load_learner('planets.pkl')

categories = ('Earth','Jupiter','Mars','Mercury','Neptune','Saturn','Uranus','Venus')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['mercury.jpg', 'venus.jpg']


iface = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
iface.launch()
