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
layer_num = 11
nb_classes = 1000
#model = torch.load("./model/alexnet-epoch5-lr_1e-05_complete.ckpt")
model = alexnet(True)
image = "./fox.jpg"
transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
imgPath = image
img = cv2.imread(image, 1)
img = np.float32(cv2.resize(img, (224, 224))) / 255
if (use_gpu):
    inp = Variable(transform(Image.open(imgPath)).unsqueeze(0).cuda(device=gpus[0]))
else:
    inp = Variable(transform(Image.open(imgPath)).unsqueeze(0))

means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]
preprocessed_img = img.copy()[: , :, ::-1]
for i in range(3):
	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
preprocessed_img = \
np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
preprocessed_img = torch.from_numpy(preprocessed_img)
preprocessed_img.unsqueeze_(0)
inp = Variable(preprocessed_img, requires_grad = True)
relu = [i for i in model.float().children()][0][12]

model1_output = torch.zeros(1, 256, 6, 6)
def fun1(m, i, o): model1_output.copy_(o.data)
h = relu.register_forward_hook(fun1)
gradinput_pool = torch.zeros(1, 256, 13, 13)
def fun2(m, i, o): gradinput_pool.copy_(i[0].data);
h2 = relu.register_backward_hook(fun2)

#forward the image
logit = model(inp)[1]


proba,pred = torch.max(logit.data,1)
goal = pred[0][0]

file_name = '../alexnet/data/image_net.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line)

print (classes[goal])
doutput = logit.data
doutput.zero_()
doutput[0][goal] = 1
h.remove()
criterion = nn.MultiLabelSoftMarginLoss()
loss= criterion(logit,Variable(doutput))
loss.backward()

activations = model1_output.numpy()[0,:]
gradients = gradinput_pool
print(model1_output)
print(gradients)
weights = np.mean( gradients.numpy(), axis=(2,3))[0,:] # should be [256]
cam = np.ones(activations.shape[1 : ], dtype = np.float32)

for i, w in enumerate(weights):
    a = activations[i, :, :]
    cam += w * a


cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224))
cam = cam - np.min(cam)
cam = cam / np.max(cam)
p1 = Image.open(image)
#cam = torch.Tensor(cam)
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
print(heatmap)
cam = heatmap + np.float32(img)
cam = cam / np.max(cam)
#cv2.imwrite("frog.jpg", np.uint8(255 * cam))
cv2.imshow('image',cam)
cv2.waitKey(0)
cv2.destroyAllWindows()
