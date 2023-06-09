import os

from torch._dynamo import variables

from dataset import get_dataset
import argparse
# from utils import *
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
from model import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

#################################################################################################
parser = argparse.ArgumentParser(description='Coupled-GAN')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--margin', default=100, type=int, help='margin')
parser.add_argument('--delta_1', default=1, type=float, help='Delta 1 HyperParameter')
parser.add_argument('--delta_2', default=1, type=float, help='Delta 2 HyperParameter')
parser.add_argument('--photo_folder', type=str,
                    default='/home/moktari/Moktari/2023/Coupled_GAN/NIR2VIS_IrisVerifier/VIS/',
                    help='path to data')
parser.add_argument('--print_folder', type=str,
                    default='//home/moktari/Moktari/2023/Coupled_GAN/NIR2VIS_IrisVerifier/NIR',
                    help='path to data')

parser.add_argument('--save_folder', type=str,
                    default='./checkpoint/',
                    help='path to save the data')

# model setup
parser.add_argument('--basenet', default='resnet18', type=str,
                    help='e.g., resnet50, resnext50, resnext101'
                         'and their wider variants, resnet50x4')
parser.add_argument('-d', '--feat_dim', default=128, type=int,
                    help='feature dimension for contrastive loss')

args = parser.parse_args()

CUDA_VISIBLE_DEVICES = 0, 1
torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


net_photo = Mapper(prenet='resnet18', outdim=args.feat_dim)
net_photo = UNet(feat_dim=args.feat_dim)
net_photo = nn.DataParallel(net_photo)
net_photo.to(device)
net_photo.train()

# net_print = Mapper(prenet='resnet18', outdim=args.feat_dim)
net_print = UNet(feat_dim=args.feat_dim)
net_print = nn.DataParallel(net_print)
net_print.to(device)
net_print.train()

disc_photo = Discriminator(in_channels=3)
disc_photo = nn.DataParallel(disc_photo)
disc_photo.to(device)
disc_photo.train()

disc_print = Discriminator(in_channels=3)
disc_print = nn.DataParallel(disc_print)
disc_print.to(device)
disc_print.train()

# for i, p in net_print.named_parameters():
#     print(i, p.size())
# exit()

optimizer_G = torch.optim.Adam(list(net_photo.parameters()) + list(net_print.parameters()), lr=1e-4, weight_decay=1e-4)
optimizer_D = torch.optim.Adam(list(disc_photo.parameters()) + list(disc_print.parameters()), lr=1e-4,
                               weight_decay=1e-4)

adversarial_loss = torch.nn.MSELoss().to(device)
L2_Norm_loss = torch.nn.MSELoss().to(device)

train_loader = get_dataset(args)

print(len(train_loader))

Tensor = torch.cuda.FloatTensor
patch = (1, 512 // 2 ** 4, 64 // 2 ** 4)

# output_dir = str(args.save_folder) + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
# os.makedirs("%s" % (output_dir), exist_ok=True)
torch.cuda.empty_cache()
for epoch in range(50):
    print(epoch)

    loss_m_d = AverageMeter()
    loss_m_g = AverageMeter()
    acc_m = AverageMeter()

    for iter, (img_photo, img_print, lbl) in enumerate(train_loader):

        # plot_tensor([img_photo[0], img_print[0]])

        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)

        img_photo, img_print, lbl = img_photo.to(device), img_print.to(device), lbl.to(device)
        # This work is for my first disc
        # valid = Variable(Tensor(np.ones((img_photo.size(0), *patch))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((img_photo.size(0), *patch))), requires_grad=False)
        valid = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False)

        # """"""""""""""""""
        # "   Generator    "
        # """"""""""""""""""

        fake_photo, y_photo = net_photo(img_photo)
        fake_print, y_print = net_print(img_print)

        # This work is for my first disc
        # pred_fake_photo = disc_photo(fake_photo, img_photo)
        # pred_fake_print = disc_print(fake_print, img_print)
        pred_fake_photo = disc_photo(fake_photo)
        pred_fake_print = disc_print(fake_print)

        loss_GAN = (adversarial_loss(pred_fake_photo, valid) + adversarial_loss(pred_fake_print, valid)) / 2
        loss_L2 = (L2_Norm_loss(fake_photo, img_photo) + L2_Norm_loss(fake_print, img_print)) / 2

        dist = ((y_photo - y_print) ** 2).sum(1)
        margin = torch.ones_like(dist, device=device) * args.margin

        loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
        loss = loss.mean() + loss_GAN * args.delta_1 + loss_L2 * args.delta_2

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        acc = (dist < args.margin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        acc_m.update(acc)

        loss_m_g.update(loss.item())

        # """"""""""""""""""
        # " Discriminator  "
        # """"""""""""""""""
        # This work is for my first disc
        # pred_real_photo = disc_photo(img_photo, img_photo)
        # pred_fake_photo = disc_photo(fake_photo.detach(), img_photo)
        #
        # pred_real_print = disc_print(img_print, img_print)
        # pred_fake_print = disc_print(fake_print.detach(), img_print)

        # This work is for my second disc
        pred_real_photo = disc_photo(img_photo)
        pred_fake_photo = disc_photo(fake_photo.detach())

        pred_real_print = disc_print(img_print)
        pred_fake_print = disc_print(fake_print.detach())

        d_loss = (
                         adversarial_loss(pred_real_print, valid)
                         + adversarial_loss(pred_real_photo, valid)
                         + adversarial_loss(pred_fake_print, fake)
                         + adversarial_loss(pred_fake_photo, fake)
                 ) / 4

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        loss_m_d.update(d_loss.item())

        if iter % 10 == 0:
            print('epoch: %02d, iter: %02d/%02d, D loss: %.4f, G loss: %.4f, acc: %.4f' % (
                epoch, iter, len(train_loader), loss_m_d.avg, loss_m_g.avg, acc_m.avg))
    state = {}
    state['net_photo'] = net_photo.state_dict()
    state['net_print'] = net_print.state_dict()
    state['optimizer'] = optimizer_G.state_dict()
    state['disc_photo'] = disc_photo.state_dict()
    state['disc_print'] = disc_print.state_dict()
    modelName = "model_unet_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2) + "_" + str(
        args.feat_dim)
    # modelName = "model_unetV2_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
    # modelName = "model_unetV3_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
    # modelName = "model_unetV4_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
    # modelName = "model_unetV5_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
    # modelName = "model_unetV6_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
    # modelName = "model_unetV1_" + str(args.margin) + "_" + str(args.delta_1) + "_" + str(args.delta_2)
    torch.save(state, args.save_folder + modelName + '.pt')
    print('\nmodel saved!\n')
