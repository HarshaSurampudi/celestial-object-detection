from fastai.vision.all import *
import gradio as gr

# Load the model
learn = load_learner('planets.pkl')

# categories for the model
# the order is important, it should match the order of the model's output
categories = ('Earth','Jupiter','Mars','Mercury','Neptune','Saturn','Uranus','Venus')

# Define a function that takes an image as input and returns a dictionary of probabilities for each category
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Input
image = gr.inputs.Image(shape=(192,192))
# Output
label = gr.outputs.Label()

# Example inputs
examples = ['mercury.jpg', 'venus.jpg']

# Define the interface
iface = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)

# share=True allows you to get a public URL. default=False
# iface.launch(share=True)
iface.launch()
