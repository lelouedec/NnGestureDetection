import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import torch.optim as optim
import random, os
import numpy as np

def load_images(path,transform):
    image_list = []
    for filename in glob.glob(path+'*.jpg'): #assuming gif
        im = Image.open(filename)
        im = transform(im.resize((225, 225), Image.BILINEAR))
        image_list.append(im)
    return image_list

def save_model(model):
    torch.save(model, "./model/random_name.ckpt")

def ComputeMean_old(imagesList, h=299, w=299):
    r,g,b = 0,0,0
    toT = transforms.ToTensor()

    for im in imagesList:
        t = toT(im)
        for e in t[0].view(-1):
            r += e
        for e in t[1].view(-1):
            g += e
        for e in t[2].view(-1):
            b += e
    return r/(len(imagesList)*h*w), g/(len(imagesList)*h*w), b/(len(imagesList)*h*w)


def ComputeStdDev_old(imagesList, mean):
    toT = transforms.ToTensor()
    r,g,b = 0,0,0
    h = len(toT(imagesList[0])[0])
    w = len(toT(imagesList[0])[0][0])
    for im in imagesList:
        t = toT(im)
        for e in t[0].view(-1):
            r += (e - mean[0])**2
        for e in t[1].view(-1):
            g += (e - mean[1])**2
        for e in t[2].view(-1):
            b += (e - mean[2])**2
    return (r/(len(imagesList)*h*w))**0.5, (g/(len(imagesList)*h*w))**0.5, (b/(len(imagesList)*h*w))**0.5

def ComputeMean(imagesList):
    r,g,b = 0,0,0
    for img in imagesList:
        rImg, gImg, bImg = img.split()
        r+=np.mean(np.array(rImg))
        g+=np.mean(np.array(gImg))
        b+=np.mean(np.array(bImg))
    return r/float(len(imagesList)), g/float(len(imagesList)), b/float(len(imagesList))

def ComputeStdDev(imagesList):
    r,g,b = 0,0,0
    for img in imagesList:
        rImg, gImg, bImg = img.split()
        r+=np.std(np.array(rImg))
        g+=np.std(np.array(gImg))
        b+=np.std(np.array(bImg))
    return r/float(len(imagesList)), g/float(len(imagesList)), b/float(len(imagesList))


def copyFeaturesParametersAlexnet(net, netBase):
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            print "copy", f
            f.weight.data = netBase.features[i].weight.data
            f.bias.data = netBase.features[i].bias.data

def copysClassifierParameterAlexnet(net, netBase):
    for i, c in enumerate(net.classifier):
        if type(c) is torch.nn.modules.linear.Linear :
            if c.weight.size() == netBase.classifier[i].weight.size():
                print "copy", c
                c.weight.data = netBase.classifier[i].weight.data
                c.bias.data = netBase.classifier[i].bias.data


def copyFeaturesParametersResnet50(net, netBase):
    net.conv1.weight.data = netBase.conv1.weight.data
    net.bn1.weight.data = netBase.bn1.weight.data
    net.bn1.bias.data = netBase.bn1.bias.data

    lLayer = [(net.layer1, netBase.layer1, 3),
              (net.layer2, netBase.layer2, 4),
              (net.layer3, netBase.layer3, 6),
              (net.layer4, netBase.layer4, 3)
             ]

    for targetLayer, rootLayer, nbC in lLayer:
        for i in range(nbC):
            targetLayer[i].conv1.weight.data = rootLayer[i].conv1.weight.data
            targetLayer[i].bn1.weight.data = rootLayer[i].bn1.weight.data
            targetLayer[i].bn1.bias.data = rootLayer[i].bn1.bias.data
            targetLayer[i].conv2.weight.data = rootLayer[i].conv2.weight.data
            targetLayer[i].bn2.weight.data = rootLayer[i].bn2.weight.data
            targetLayer[i].bn2.bias.data = rootLayer[i].bn2.bias.data
            targetLayer[i].conv3.weight.data = rootLayer[i].conv3.weight.data
            targetLayer[i].bn3.weight.data = rootLayer[i].bn3.weight.data
            targetLayer[i].bn3.bias.data = rootLayer[i].bn3.bias.data
        targetLayer[0].downsample[0].weight.data = rootLayer[0].downsample[0].weight.data
        targetLayer[0].downsample[1].weight.data = rootLayer[0].downsample[1].weight.data
        targetLayer[0].downsample[1].bias.data = rootLayer[0].downsample[1].bias.data


def trainclassifier(net, optimizer, criterion, batchSize, dImgTrain, imSize, imageTransform, dClassImgTrain):
    net.train()
    # shuffle images name
    lImgName = dImgTrain.keys()
    random.shuffle(lImgName)
    # Split the whole list into sublist sizeof batch_size
    for subListImgName in [lImgName[i:i+batchSize] for i in range(0, len(lImgName), batchSize)][:-1]:
        # transform images into tensor
        inputs = torch.Tensor(batchSize, 3, imSize, imSize).cuda()
        for i, imgName in enumerate(subListImgName): inputs[i] = imageTransform(dImgTrain[imgName])
        inputs = Variable(inputs)
        # list class of the sublist images
        lab = Variable(torch.LongTensor([dClassImgTrain[imgName] for imgName in subListImgName]).cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, _ = net(inputs)
        loss = criterion(outputs, lab)
        loss.backward()
        optimizer.step()


def testClassifier(net, dImgTest, imSize, imageTransform):
    net.eval()
    nbCorrect = 0
    for imgName in dImgTest:
        inp = torch.Tensor(1,3, imSize, imSize).cuda()
        inp[0] = imageTransform(dImgTest[imgName])

        outputs, _ = net(Variable(inp))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        nbCorrect+= (predicted[0][0] == dClassU3Train[imgName[:5]])

    print "test : #Correct "+str(nbCorrect)+" on "+str(len(pathImgU3Test))+" ("+str(round(float(nbCorrect)/float(len(pathImgU3Test))*100, 1))+"%)"
