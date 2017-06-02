# NnGestureDetection

Repository dedicated to an internship project about Gesture Recognition.

what you can do now :
- train a alexnet network from scratch
- test it for the validation dataset and display the percentage of Accuracy
- visualize pictures with the classes predicted for them
- copy features from a network to an other
- create a scheduler for the learning rate

Choose to execute on GPU or CPU from the commande line ( --gpu 0/1)

Grad-cam.py : It is the implementation of the paper : https://arxiv.org/abs/1610.02391 in pytorch
(Fixed thanks to the implementation of https://github.com/jacobgil/pytorch-grad-cam)

Can be used as python grad-cam.py <aimed layer> <mode> <image>
