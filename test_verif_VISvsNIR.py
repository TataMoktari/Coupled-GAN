import os

from dataset import get_dataset
import argparse
# from utils import *
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from utils import *
from model import *


#################################################################################################
def comp_bnfeatures(self):
    # extract batchnorm features
    feature = []
    for i in range(len(self.bn_hook)):
        feature.append(self.bn_hook[i].delta_mean)
        feature.append(self.bn_hook[i].delta_var)

    feature = torch.cat(feature, 1)
    return feature


def parameters_fc(self):
    return [self.fc1.weight] + [self.fc1.bias] + [self.fc2.weight] + [self.fc2.bias]


def parameters_net(self):
    return self.basemodel.parameters()


#################################################################################################


parser = argparse.ArgumentParser(description='Coupled-GAN')
parser.add_argument('--batch_size', default=96, type=int, help='batch size')
parser.add_argument('--margin', default=5, type=int, help='margin')
parser.add_argument('--delta_1', default=1, type=float, help='Delta 1 HyperParameter')
parser.add_argument('--delta_2', default=1, type=float, help='Delta 2 HyperParameter')
parser.add_argument('--photo_folder', type=str,
                    default='../NIR2VIS_IrisVerifier/VIS/',
                    help='path to data')
parser.add_argument('--print_folder', type=str,
                    default='../NIR2VIS_IrisVerifier/NIR/',
                    help='path to data')

parser.add_argument('--save_folder', type=str,
                    default='./checkpoint/',
                    help='path to save the data')

# model setup
parser.add_argument('--basenet', default='unet', type=str,
                    help='e.g., resnet50, resnext50, resnext101'
                         'and their wider variants, resnet50x4')
parser.add_argument('--net', default='unetV1', type=str)
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')

args = parser.parse_args()

device = torch.device('cuda:0')

output_dir = "data/" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2) + "_" + str(args.feat_dim)
os.makedirs("%s" % (output_dir), exist_ok=True)
state = torch.load('./%s/model_unet_%s_%s_%s_%s.pt' % (
args.save_folder, str(args.margin), str(args.delta_1), str(args.delta_2), str(args.feat_dim)))


if args.net == "unetV1":
    net_photo = UNet(feat_dim=args.feat_dim)

net_photo.to(device)
net_photo.load_state_dict(state['net_photo'])
net_photo.eval()


if args.net == "unetV1":
    net_print = UNet(feat_dim=args.feat_dim)

net_print.to(device)
net_print.load_state_dict(state['net_print'])
net_print.eval()

train_loader = get_dataset(args)

print(len(train_loader))

dist_l = []
lbl_l = []
for iter, (img_photo, img_print, lbl) in enumerate(train_loader):
    print(iter)
    bs = img_photo.size(0)
    lbl = lbl.type(torch.float)

    img_photo, img_print, lbl = img_photo.to(device), img_print.to(device), lbl.to(device)

    _, y_photo = net_photo(img_photo)
    _, y_print = net_print(img_print)

    dist = ((y_photo - y_print) ** 2).sum(1)
    dist_l.append(dist.data)
    lbl_l.append((1 - lbl).data)

dist = torch.cat(dist_l, 0)
lbl = torch.cat(lbl_l, 0)
dist = dist.cpu().detach().numpy()
lbl = lbl.cpu().detach().numpy()

import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(lbl, dist)
roc_auc = metrics.auc(fpr, tpr)

import numpy as np

np.save('%s/pr_ph_lbl_test.npy' % (output_dir), lbl)
np.save('%s/pr_ph_dist_test.npy' % (output_dir), dist)

# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
