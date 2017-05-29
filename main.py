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

use_gpu = 0




def main(argv):
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["gpu="])
   except getopt.GetoptError:
      print ("correct syntax : main.py --gpu <0/1 use_gpu>")
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ("main.py --gpu <0/1 use_gpu>")
         sys.exit(0)
      elif opt in ("--gpu"):
         global use_gpu
         use_gpu = int(arg)
         utils.use_gpu = int(arg)
   print ('Use gpu ? ', use_gpu)

if __name__ == '__main__':
    main(sys.argv[1:])
    #train_from_scratch("alexnet")
    #test_network("./model/resnet34-epoch5-lr_1e-05_complete.ckpt")
    #print ("test class 1 ")
    #test_image("./dataset/val/1/","./model/alexnet-epoch5-lr_0.00000001_complete.ckpt")
    #print ("test class 2")
    #test_image("./dataset/val/2/","./model/alexnet-epoch5-lr_0.00000001_complete.ckpt")
    #print("test class 3")
    #test_image("./dataset/val/3/","./model/alexnet-epoch5-lr_0.00000001_complete.ckpt")
    #print("test no a hand")
    #test_image("./dataset/val/ImagesDiversTest/","./model/alexnet-epoch5-lr_0.00000001_complete.ckpt")
    #data_collect()
    display_features(torch.load("./model/resnet34-epoch5-lr_1e-05_complete.ckpt"))
