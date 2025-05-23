import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')



# class TempModel(nn.Module):
#     def __init__(self):
#         super(TempModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 5, (3, 3))

#     def forward(self, inp):
#         return self.conv1(inp)

class PlantDiseaseModel(nn.Module):
    def __init__(self):
        super(PlantDiseaseModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as needed
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(32 * 112 * 112, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Adjust the output for your number of classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense_layers(x)
        return x


# model = PlantDiseaseModel()
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()

model = PlantDiseaseModel()
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"), strict=False)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# def predict():
#     try:
#         # Get the uploaded image
#         image = request.files['image']
#         img = Image.open(image)
#         img = transform(img)

#         # Make a prediction
#         with torch.no_grad():
#             output = model(img.unsqueeze(0))
#             predicted_class = torch.argmax(output)

#         return jsonify({'prediction': predicted_class.item()})

#     except Exception as e:
#         return jsonify({'error': str(e)})

def predict(image_path):
    try:
        # Open the image from the provided path
        img = Image.open(image_path)
        img = transform(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output)

        return predicted_class.item()

    except Exception as e:
        return {'error': str(e)}


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')
    #comment

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        # Call the updated predict function with file_path as an argument
        pred = predict(file_path)
        
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        return render_template(
            'submit.html', title=title, desc=description, prevent=prevent,
            image_url=image_url, pred=pred, sname=supplement_name, 
            simage=supplement_image_url, buy_link=supplement_buy_link
        )
# @app.route('/submit', methods=['GET', 'POST'])
# def submit():
#     if request.method == 'POST':
#         image = request.files['image']
#         filename = image.filename
#         file_path = os.path.join('static/uploads', filename)
#         image.save(file_path)
#         print(file_path)
#         pred = predict(file_path)  # Call the predict function here
#         title = disease_info['disease_name'][pred]
#         description = disease_info['description'][pred]
#         prevent = disease_info['Possible Steps'][pred]
#         image_url = disease_info['image_url'][pred]
#         supplement_name = supplement_info['supplement name'][pred]
#         supplement_image_url = supplement_info['supplement image'][pred]
#         supplement_buy_link = supplement_info['buy link'][pred]
#         return render_template('submit.html', title=title, desc=description, prevent=prevent,
#                                image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
