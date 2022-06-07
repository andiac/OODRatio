import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from imageio import imread
from torch import Tensor
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode

from utils import OE_Dataset, RandomSizeCrop

"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.
Set root to point to the Train/Test folders.

Refer to: https://github.com/Aftaab99/PyTorchImageClassifier/blob/master/dataloader.py 
"""

# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

    # The init method is called when this class will be instantiated.
    def __init__(self, root):
        Images, Y = [], []
        folders = os.listdir(root)

        for folder in folders:
            folder_path = os.path.join(root, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    Images.append(np.array(imread(img_path)))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    # Some images in the dataset are damaged
                    print("File {}/{} is broken".format(folder, ims))
        data = [(x, y) for x, y in zip(Images, Y)]
        self.data = data

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    # The Dataloader is a generator that repeatedly calls the getitem method.
    # getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        img = self.data[index][0]

        # 8 bit images. Scale between [0,1]. This helps speed up our training
        img = img.reshape(28, 28) / 255.0

        # Input for Conv2D should be Channels x Height x Width
        img_tensor = Tensor(img).view(1, 28, 28).float()
        label = self.data[index][1]
        return (img_tensor, label)




def get_dataloaders(in_dist_dataset, batch_size, eval_batch_size, data_dir, train_kwargs, kwargs):

    # data_dir = "../data"
    download = False

    transform_oe = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28))
    ])

    transform_omniglot = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])

    transform_crop_resize = transforms.Compose([
        RandomSizeCrop([14, 20, 25, 26, 27]),
        transforms.Resize(28, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])

    # MNIST: 60000/10000 28x28
    mnist_train_dl = DataLoader(
                        datasets.MNIST(data_dir, train=True, download=download, transform=transforms.ToTensor()),
                        batch_size=batch_size, shuffle=True, **train_kwargs
                    )
    mnist_test_dl = DataLoader(
                        datasets.MNIST(data_dir, train=False, download=download, transform=transforms.ToTensor()),
                        batch_size=eval_batch_size, shuffle=False, **kwargs
                    )
    # FashionMNIST: 60000/10000 28x28
    fmnist_train_dl = DataLoader(
                        datasets.FashionMNIST(data_dir, train=True, download=download, transform=transforms.ToTensor()),
                        batch_size=batch_size, shuffle=True, **train_kwargs
                    )
    fmnist_test_dl = DataLoader(
                        datasets.FashionMNIST(data_dir, train=False, download=download, transform=transforms.ToTensor()),
                        batch_size=eval_batch_size, shuffle=False, **kwargs
                    )
    # KMNIST: 60000/10000 28x28
    kmnist_test_dl = DataLoader(
                        datasets.KMNIST(data_dir, train=False, download=download, transform=transforms.ToTensor()),
                        batch_size=eval_batch_size, shuffle=False, **kwargs
                    )
    # Omniglot:
    omni_test_dl = DataLoader(
                        datasets.Omniglot(data_dir, background=True, download=download, transform=transform_omniglot),
                        batch_size=eval_batch_size, shuffle=False, **kwargs
                    )
    # notMNIST: 18500/469 28x28
    notmnist_test_dl = DataLoader(
                        notMNIST(os.path.join(data_dir, 'notMNIST/Train')),
                        batch_size=eval_batch_size, shuffle=False, **kwargs
                    )
    # OOD Outliner Exposure
    oe_dataloader = DataLoader(
                        OE_Dataset(os.path.join(data_dir, "300K_random_images.npy"), transform=transform_oe),
                        batch_size=batch_size, shuffle=True, **train_kwargs
                    )

    if in_dist_dataset == "MNIST":
        train_loader = mnist_train_dl
        train_loader_2 = DataLoader(
                        datasets.MNIST(data_dir, train=True, download=download, transform=transforms.ToTensor()),
                        batch_size=batch_size, shuffle=True, **train_kwargs
                    )
        test_loader = mnist_test_dl
        crop_resize_dl = DataLoader(
                                datasets.MNIST(data_dir, train=True, download=download, transform=transform_crop_resize),
                                batch_size=batch_size, shuffle=True, **train_kwargs)
        return train_loader, train_loader_2, test_loader, oe_dataloader, fmnist_test_dl, kmnist_test_dl, notmnist_test_dl, omni_test_dl, crop_resize_dl
    elif in_dist_dataset == "FashionMNIST":
        train_loader = fmnist_train_dl
        train_loader_2 = DataLoader(
                        datasets.FashionMNIST(data_dir, train=True, download=download, transform=transforms.ToTensor()),
                        batch_size=batch_size, shuffle=True, **train_kwargs
                    )
        test_loader = fmnist_test_dl
        crop_resize_dl = DataLoader(
                                 datasets.FashionMNIST(data_dir, train=True, download=download, transform=transform_crop_resize),
                                 batch_size=batch_size, shuffle=True, **train_kwargs)
        return train_loader, train_loader_2, test_loader, oe_dataloader, mnist_test_dl, kmnist_test_dl, notmnist_test_dl, omni_test_dl, crop_resize_dl
    else:
        raise NotImplementedError
