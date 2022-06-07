import os
import wandb
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import InterpolationMode

lsoftmax = torch.nn.LogSoftmax(dim=1)

from sklearn.metrics import roc_auc_score

from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNetDrop18
from models.densenet import DenseNet3
from models.wide_resnet import WideResNet2810
from utils import OE_Dataset

class RandomSizeCrop(torch.nn.Module):
    def __init__(self, sizes, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.randomcrops = []
        for size in sizes:
            self.randomcrops.append(transforms.RandomCrop(size, padding, pad_if_needed, fill, padding_mode))
        
    def forward(self, img):
        l = len(self.randomcrops)
        idx = random.randrange(l)
        return self.randomcrops[idx](img)

parser = argparse.ArgumentParser('Outlier Exposure by Classification')
parser.add_argument('--data-dir', type=str, default="../data")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--optimizer', type=str, default="SGD", choices=['SGD', 'Adam'])
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--noise-std', type=float, default=0.0)
parser.add_argument('--anneal-alpha-init', type=float, default=1.0)
parser.add_argument('--anneal-gamma', type=float, default=0.142857)
parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'DenseNet3', 'ResNetDrop18', 'WideResNet2810'])

args = parser.parse_args()

batch_size = args.batch_size
lr = args.lr
EPOCH = args.epochs
anneal_alpha = args.anneal_alpha_init

val_batch_size = 4096
num_workers = 4
loss = torch.nn.CrossEntropyLoss(reduction='mean')
lsoftmax = torch.nn.LogSoftmax(dim=1)
train_kwargs = {'num_workers':num_workers, 'pin_memory':True, 'drop_last':True}  # drop last True?
kwargs = {'num_workers':num_workers, 'pin_memory':True, 'drop_last':False}  # drop last True?
device = torch.device('cuda')

config_dict = {
  "in_dist": "cifar100",
  "learning_rate": lr,
  "epochs": EPOCH,
  "batch_size": batch_size,
  "model_name": args.model,
  "weight_decay": args.weight_decay,
  "noise_std": args.noise_std,
  "anneal_alpha_init": args.anneal_alpha_init,
  "anneal_gamma": args.anneal_gamma,
  "optimizer": args.optimizer
}

# anonymous
# wandb.init(...)

net = eval(args.model)(out_dim=2)
net.to(device)
if args.optimizer == "SGD":
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_crop_resize = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(args.crop_size), 
    RandomSizeCrop([20, 22, 24, 26, 28, 30]),
    transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=transform_train), batch_size=batch_size, shuffle=True, **train_kwargs)
train_loader_2 = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=transform_train), batch_size=batch_size, shuffle=True, **train_kwargs)

ood_train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=transform_crop_resize), batch_size=batch_size, shuffle=True, **train_kwargs)

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False, 
                transform=transform_test), batch_size=batch_size, shuffle=False, **kwargs)
svhn_loader  = torch.utils.data.DataLoader(datasets.SVHN(args.data_dir, split="test", 
                transform=transform_test, download=True), batch_size=batch_size, shuffle=False, **kwargs)

cifar10_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                transform=transform_test, download=True), batch_size=batch_size, shuffle=False, **kwargs)
celeba_loader = torch.utils.data.DataLoader(datasets.CelebA(root=args.data_dir, split="test", transform=transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()]), download=False), batch_size=batch_size, shuffle=False, **kwargs)

lsun_loader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(args.data_dir, 'LSUN_resize'), transform=transform_test), batch_size=batch_size, shuffle=False, **kwargs)

writer = SummaryWriter(log_dir=f"./logs/cifar100_log/{args.model}_{batch_size}_{lr}_{args.noise_std}_{anneal_alpha}_{args.anneal_gamma}")

for e in range(EPOCH):
    net.train()
    train_loss = 0.0
    train_N = 0
    for (x_in, _), (x_in_2, _), (x_out, _) in zip(train_loader, train_loader_2, ood_train_loader):
        optimizer.zero_grad()
        x_in    = x_in.to(device)
        x_in_2  = x_in_2.to(device)
        x_out   = x_out.to(device)

        assert x_in.shape[0] == x_out.shape[0]
        # annealing
        anneal_mask = torch.bernoulli(torch.ones(x_in.shape[0], device=device) * anneal_alpha).unsqueeze(1).repeat(1, 3*32*32).view(x_in.shape[0], 3, 32, 32)
        x_out = anneal_mask * x_in_2 + (1.0 - anneal_mask) * x_out
        
        comb_x = torch.cat((x_in, x_out), dim=0)
        train_N += comb_x.shape[0]

        y_in  = torch.zeros(x_in.shape[0],  dtype=torch.long).to(device)
        y_out = torch.ones (x_out.shape[0], dtype=torch.long).to(device)
        comb_y = torch.cat((y_in, y_out), dim=0)

        # apply noise
        comb_x = comb_x + torch.randn(comb_x.shape, device=device) * args.noise_std
        logits = net.forward(comb_x)
        l = loss(logits, comb_y)
        l.backward()
        train_loss += comb_y.shape[0] * l.item()
        optimizer.step()

    anneal_alpha *= args.anneal_gamma

    def get_scores(loader):
        scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = net.forward(x)
                # lprobs = lsoftmax(logits)
                score = logits[:, 1] - logits[:, 0]
                scores.append(score.detach().cpu().numpy())
        scores = np.concatenate(scores)
        return scores

    net.eval()
    test_scores = get_scores(test_loader)[:10000]
    svhn_scores = get_scores(svhn_loader)[:10000]
    cifar10_scores = get_scores(cifar10_loader)[:10000]
    celeba_scores = get_scores(celeba_loader)[:10000]
    lsun_scores = get_scores(lsun_loader)[:10000]

    svhn_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, svhn_scores)))
    cifar10_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, cifar10_scores)))
    celeba_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, celeba_scores)))
    lsun_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, lsun_scores)))

    print(f"epoch {e}: train_loss: {train_loss / train_N:.4f}, svhn_AUROC: {svhn_AUROC: .4f}, cifar10_AUROC: {cifar10_AUROC:.4f}, celeba_AUROC: {celeba_AUROC:.4f}, lsun_AUROC: {lsun_AUROC:.4f}")

    writer.add_scalar("train_loss", train_loss / train_N, e)
    writer.add_scalar("svhn_AUROC", svhn_AUROC, e)
    writer.add_scalar("cifar10_AUROC", cifar10_AUROC, e)
    writer.add_scalar("celeba_AUROC", celeba_AUROC, e)
    writer.add_scalar("lsun_AUROC", lsun_AUROC, e)

    wandb.log({"train_loss": train_loss / train_N,
               "svhn_AUROC": svhn_AUROC,
               "cifar10_AUROC": cifar10_AUROC,
               "celeba_AUROC": celeba_AUROC,
               "lsun_AUROC": lsun_AUROC,
               "test_scores": test_scores,
               "svhn_scores": svhn_scores,
               "cifar10_scores": cifar10_scores,
               "celeba_scores": celeba_scores,
               "lsun_scores": lsun_scores
               })

writer.close()
wandb.finish()

