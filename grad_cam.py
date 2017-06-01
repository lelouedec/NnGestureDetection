from __future__ import print_function, division

from alexnet  import *
from resnet import *
from resnet import BasicBlock
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import pickle, os, glob
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import os
import sys, getopt
from Utils import *
import Utils as utils
import cv2

use_gpu = 0
layer_num = int(sys.argv[1])
nb_classes = 1000
#model = torch.load("./model/alexnet-epoch5-lr_1e-05_complete.ckpt")
if(sys.argv[2] == "alexnet"):
    model = alexnet(True)
elif(sys.argv[2] == "vgg"):
    model = models.vgg19(pretrained=True)
image = sys.argv[3]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = cv2.imread(image, 1)
img_inp = np.float32(cv2.resize(img, (224, 224)))/255
if (use_gpu):
    inpu = Variable(transform(img_inp).unsqueeze(0).cuda(device=gpus[0]))
else:
    inpu = Variable(transform(img_inp).unsqueeze(0), requires_grad = True)

means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]

preprocessed_img = img_inp.copy()[: , :, ::-1]
for i in range(3):
	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

preprocessed_img =  np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
preprocessed_img = torch.from_numpy(preprocessed_img)
inp = Variable(preprocessed_img.unsqueeze(0), requires_grad = True)
print(inp.data)
if(torch.equal(inp.data,inpu.data)):
    print("equals")

outputs = []#outputs of each layers
gradients = []#gradinputs of each layers
def grad(g):gradients.append(g)
model.eval()
for module in model.features._modules.items():
    inp = module[1](inp)
    if (int(module[0]) == layer_num) :
        inp.register_hook(grad)
        outputs += [inp]


inp = inp.view(inp.size(0), -1)
output = model.classifier(inp)#end of network

index = np.argmax(output.data.numpy())
print(index)
one_hot = output.data
one_hot.zero_()
one_hot[0][index] = 1
one_hot = Variable(one_hot,requires_grad = True)
one_hot = torch.sum(one_hot * output)

model.features.zero_grad()
model.classifier.zero_grad()
one_hot.backward(retain_graph=True)
grad_input = gradients[-1].data.numpy()
activations = outputs[-1].data.numpy()[0,:]

weights = np.mean( grad_input, axis=(2,3))[0,:]
cam = np.ones(activations.shape[1 : ], dtype = np.float32)

for i, w in enumerate(weights):
    a = activations[i, :, :]
    cam += w * a


cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam - np.min(cam)
cam = cam / np.max(cam)
p1 = Image.open(image)
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
cam = heatmap + np.float32(img_inp)
cam = cam / np.max(cam)
#cv2.imwrite("frog.jpg", np.uint8(255 * cam))
cv2.imshow('image',np.uint8(255 * cam))
cv2.waitKey(0)
cv2.destroyAllWindows()
