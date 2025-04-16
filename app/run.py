import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO
import base64
import io
import time
from collections import OrderedDict
import json

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.utils as pu

from flask import Flask
from flask import render_template, request

import sys
sys.path.insert(0, 'c:/Users/Yash/Desktop/quick-draw-image-recognition-master')
import image_utils as image_utils
from image_utils import crop_image, normalize_image, convert_to_rgb, convert_to_np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

label_dict = {0:'cannon',1:'eye', 2:'face', 3:'nail', 4:'pear',
              5:'piano',6:'radio', 7:'spider', 8:'star', 9:'sword'}

def load_model(filepath = 'checkpoint.pth'):
    """
    Function loads the model from checkpoint.

    INPUT:
        filepath - path for the saved model

    OUTPUT:
        model - loaded pytorch model
    """

    print("Loading model from {} \n".format(filepath))

    checkpoint = torch.load(filepath)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_sizes = checkpoint['hidden_layers']
    dropout = checkpoint['dropout']
    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                          ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                          ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(hidden_sizes[2], output_size))]))
    model.load_state_dict(checkpoint['state_dict'])

    return model

def get_prediction(model, input):
    """
    Function to get prediction (label of class with the greatest probability).

    INPUT:
        model - pytorch model
        input - (numpy) input vector

    OUTPUT:
        label - predicted class label
        label_name - name of predicted class
    """
    # Convert input to tensor
    input = torch.from_numpy(input).float()
    input = input.resize_(1, 784)

    with torch.no_grad():
        logits = model.forward(input)
    ps = F.softmax(logits, dim=1)

    preds = ps.numpy()

    label = np.argmax(preds)
    label_name = label_dict[label] 

    return label, label_name, preds

def view_classify(img, preds):
    """
    Function for viewing an image and it's predicted classes
    with matplotlib.

    INPUT:
        img - (numpy) image file
        preds - (numpy) predicted probabilities for each class
    """
    preds = preds.squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), preds)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(['cannon','eye', 'face', 'nail', 'pear','piano','radio','spider','star','sword'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    ts = time.time()
    plt.savefig('prediction' + str(ts) + '.png')

app = Flask(__name__)

# load model
model = load_model()
model.eval() # set to evaluation

# index webpage receives user input for the model
@app.route('/')
@app.route('/index')
def index():
    # render web page
    return render_template('index.html')

@app.route('/go/<dataURL>')
def pred(dataURL):
    """
    Render prediction result.
    """

    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')

    image_b64_str = dataURL
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)

    ts = time.time()

    img = img.convert("RGBA")

    # preprocess the image for the model
    image_cropped = crop_image(img) 
    image_normalized = normalize_image(image_cropped)

    img_rgb = convert_to_rgb(image_normalized)

    image_np = convert_to_np(img_rgb)

    label, label_num, preds = get_prediction(model, image_np)
    print("This is a {}".format(label_num))

    view_classify(image_np, preds)

    # create plotly visualization
    graphs = [
        {
            'data': [
                go.Bar(
                        x = preds.ravel().tolist(),
                        y = list(label_dict.values()),
                        orientation = 'h')
            ],

            'layout': {
                'title': 'Class Probabilities',
                'yaxis': {
                    'title': "Classes"
                },
                'xaxis': {
                    'title': "Probability",
                }
            }
        }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls= pu.PlotlyJSONEncoder)

    return render_template(
        'hook.html',
        result = label_num, 
        ids=ids, 
        graphJSON=graphJSON, 
        dataURL = dataURL 
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
