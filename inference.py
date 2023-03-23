import subprocess
subprocess.call(['pip', 'install', 'smdebug'])
#subprocess.call(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])
subprocess.run(['pip', 'install', 'botocore==1.23.24'])
#pip.main(['install', 'sagemaker==1.72.0'])

import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import numpy as np
import cv2


def model_fn(model_dir):
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))


def predict_fn(input_object, model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object=transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction