import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import torch
from utils import *


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ContrastiveDataset(Dataset):
    def __init__(self, morph_dataset, photo_dataset, positive_prob=0.5):
        super().__init__()
        self.print = morph_dataset
        self.photo = photo_dataset
        self.positive_prob = positive_prob

        print(len(self.photo))  ### any random folder
        print(len(self.print))

        self.positive_h = {}
        self.negative_h = {}

        for i in range(len(self.print)):
            # contruct the positive pair correspondence
            img_address = self.print.imgs[i][0]
            id = img_address.split('/')[-2]
            if id in self.positive_h:
                self.positive_h[id].append(i)
            else:
                self.positive_h[id] = [i]
            # construct the negative pair correspondence
            for j in range(len(self.photo.imgs)):
                profile_address = self.photo.imgs[j][0]
                if id in profile_address:
                    if id in self.negative_h:
                        self.negative_h[id].append(j)
                    else:
                        self.negative_h[id] = [j]

    def __getitem__(self, index):
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob
        img_0, label_0 = self.print[index]

        if same_class:  # pick a positive sample
            img_address = self.print.imgs[index][0]
            id = img_address.split('/')[-2]
            idx_positive = self.positive_h[id]
            rnd_idx = random.randint(0, len(idx_positive) - 1)
            idx_positive = idx_positive[rnd_idx]
            img_1, label_1 = self.print[idx_positive]
        else:
            img_address = self.print.imgs[index][0]
            id = img_address.split('/')[-2]
            idx_neg = self.negative_h[id]
            rnd_idx = random.randint(0, len(idx_neg) - 1)
            idx_neg = idx_neg[rnd_idx]
            img_1, label_1 = self.photo[idx_neg]

        return img_0, img_1, same_class

    def __len__(self):
        # return min(len(self.morph), len(self.photo))
        return len(self.print)


def fixed_image_standardization(image_tensor):
    # processed_tensor = (image_tensor - 127.5) / 128.0
    processed_tensor = (image_tensor - .5) / .5
    return processed_tensor


def get_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    photo_dataset = datasets.ImageFolder(
        args.photo_folder,
        transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))

    print_dataset = datasets.ImageFolder(
        args.print_folder,
        transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]))

    train_loader = torch.utils.data.DataLoader(
        ContrastiveDataset(print_dataset, photo_dataset), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return train_loader
