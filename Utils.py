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

def ComputeMean(imagesList, depth=256):
    """
    Compute the mean value of each RGB channel of a set of images
    """
    r,g,b,i = 0.0, 0.0, 0.0, 0.0
    for img in imagesList:
        try:
            rImg, gImg, bImg = img.split()
            r+=np.mean(np.array(rImg))
            g+=np.mean(np.array(gImg))
            b+=np.mean(np.array(bImg))
            i+=1
        except:
            pass
    return r/i/depth, g/i/depth, b/i/depth

def ComputeStdDev(imagesList, depth=256):
    """
    Compute the standard deviation value of each RGB channel of a set of images
    """
    r,g,b,i = 0.0, 0.0, 0.0, 0.0
    for img in imagesList:
        try:
            rImg, gImg, bImg = img.split()
            r+=np.std(np.array(rImg))
            g+=np.std(np.array(gImg))
            b+=np.std(np.array(bImg))
            i+=1
        except:
            pass
    return r/i/depth, g/i/depth, b/i/depth
