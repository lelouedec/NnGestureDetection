from __future__ import print_function, division

from alexnet  import *
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
import time
import copy
import os
use_gpu = 0
plt.ion()   # interactive mode
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #selecting the graphic processor
cudnn.benchmark = True #-- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                       #-- If this is set to false, uses some in-built heuristics that might not always be fastest.

cudnn.fastest = True #-- this is like the :fastest() mode for the Convolution modules,
                     #-- simply picks the fastest convolution algorithm, rather than tuning for workspace size
imSize = 225
batchSize = 32
nb_epoch = 50


data_dir	 = "./dataset/"

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])for x in ['train', 'val']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dset_loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[dset_classes[x] for x in classes])

def copyFeaturesParametersAlexnet(net, netBase):
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            print ("copy", f)
            f.weight.data = netBase.features[i].weight.data
            f.bias.data = netBase.features[i].bias.data

def train_model(model, criterion, optimizer,  num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)[1]

            	loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                print('epoch %d running_loss: %.3f, loss: %0.3f' % (epoch+1,  running_loss / 10,loss.data[0]))
		if (running_loss > 1000000.0):
			running_loss = 0

            epoch_loss = running_loss / dset_sizes[phase]
            best_model = copy.deepcopy(model)

        test_model(model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return best_model

#Function for displaying prediction for images
def visualize_model(model, num_images=3):
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(dset_loaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)[1]
        _, preds = torch.max(outputs, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[labels.data[j]]))
            imshow(inputs.cpu().data[j])
            if images_so_far == num_images:
                return

def test_model(model):
    correct = 0
    total = 0
    for data in dset_loaders['val']:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    #we use a pretrained model of Alexnet and copy only features into our model
    alexnextmodel = alexnet(True)
    alexTunedClassifier = AlexNet()
    copyFeaturesParametersAlexnet(alexTunedClassifier, alexnextmodel)

    if use_gpu:
     	alexTunedClassifier.cuda()

    criterion = nn.CrossEntropyLoss()
    #we dont train last layers
    optimizer=optim.SGD([{'params': alexTunedClassifier.classifier.parameters()},
                         {'params': alexTunedClassifier.features.parameters(), 'lr': 0.0}
                        ], lr=0.001, momentum=0.9)

    model2 = train_model(alexTunedClassifier, criterion, optimizer, num_epochs=5)
    torch.save(model2, "./model/alexnet-epoch5-lr_0.001_notcomplete.ckpt")
    visualize_model(model2,10)

    model2 = torch.load( "./model/alexnet-epoch5-lr_0.001_notcomplete.ckpt")
    visualize_model(model2,10)


    #we train everything but with a lower learning rate
    optimizer=optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)

    model2 = train_model(alexTunedClassifier, criterion, optimizer, num_epochs=5)
    torch.save(model2, "./model/alexnet-epoch5-lr_0.001_complete.ckpt")

    #we reduce again the learning rate
    optimizer=optim.SGD(model2.parameters(), lr=0.0001, momentum=0.9)
    model2 = train_model(alexTunedClassifier, criterion, optimizer,
                           num_epochs=5)
    torch.save(model2, "./model/alexnet-epoch5-lr_0.0001_complete.ckpt")
    visualize_model(model2,3)
