from os import listdir
from random import choice
import torch as torch
import numpy as np
import torch.nn as nn

ext2conttype = {"jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif"}

def content_type(filename):
    return ext2conttype[filename[filename.rfind(".")+1:].lower()]

def isimage(filename):
    """true if the filename's extension is in the content-type lookup"""
    filename = filename.lower()
    return filename[filename.rfind(".")+1:] in ext2conttype

def random_file(dir):
    """returns the filename of a randomly chosen image in dir"""
    images = [f for f in listdir(dir) if isimage(f)]
    return choice(images)

def copyFeaturesParametersAlexnet(net, netBase):
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            print ("copy", f)
            f.weight.data = netBase.features[i].weight.data
            f.bias.data = netBase.features[i].bias.data
    print ("network copied")

def copyFeaturesParametersResnet(net, netBase, nbBlock1, nbBlock2, nbBlock3, nbBlock4, typeBlock="Bottleneck"):
    """
    Copy all parameters of a Resnet model from a pretrained model (except for the last fully connected layer)
    typeBlock == "BasicBlock" for resnet18 and resnet34 or "Bottleneck" for resnet50, resnet101 and resnet152
    resnet18: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 2, 2, 2, 2
    resnet34: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 4, 6, 3
    resnet50: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 4, 6, 3
    resnet101: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 4, 23, 3
    resnet152: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 8, 36, 3
    see model/resnet.py for more informations about the model
    """

    if typeBlock not in ["BasicBlock", "Bottleneck"]:
        print 'error in the block name, choose "BasicBlock", "Bottleneck"'
        return

    print "copy net.conv1", net.conv1
    net.conv1.weight.data = netBase.conv1.weight.data
    print "copy net.bn1", net.bn1
    net.bn1.weight.data = netBase.bn1.weight.data
    net.bn1.bias.data = netBase.bn1.bias.data

    lLayer = [("layer1", net.layer1, netBase.layer1, nbBlock1),
              ("layer2", net.layer2, netBase.layer2, nbBlock2),
              ("layer3", net.layer3, netBase.layer3, nbBlock3),
              ("layer4", net.layer4, netBase.layer4, nbBlock4)
             ]
    print("type block " +typeBlock)
    if typeBlock == "BasicBlock":
        for layerName, targetLayer, rootLayer, nbC in lLayer:
            print "copy", layerName, rootLayer
            for i in range(nbC):
                targetLayer[i].conv1.weight.data = rootLayer[i].conv1.weight.data
                targetLayer[i].bn1.weight.data = rootLayer[i].bn1.weight.data
                targetLayer[i].bn1.bias.data = rootLayer[i].bn1.bias.data
                targetLayer[i].conv2.weight.data = rootLayer[i].conv2.weight.data
                targetLayer[i].bn2.weight.data = rootLayer[i].bn2.weight.data
                targetLayer[i].bn2.bias.data = rootLayer[i].bn2.bias.data
            if targetLayer[0].downsample:
                targetLayer[0].downsample[0].weight.data = rootLayer[0].downsample[0].weight.data
                targetLayer[0].downsample[1].weight.data = rootLayer[0].downsample[1].weight.data
                targetLayer[0].downsample[1].bias.data = rootLayer[0].downsample[1].bias.data

    elif typeBlock == "Bottleneck":
        for layerName, targetLayer, rootLayer, nbC in lLayer:
            print "copy", layerName, rootLayer
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
