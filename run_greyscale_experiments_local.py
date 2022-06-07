import os
import wandb
import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score

from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils import OE_Dataset
import greyscale_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Outlier Exposure by Classification')
    parser.add_argument('--data-dir', type=str, default="../data")
    parser.add_argument('--in_dist_dataset', choices=['MNIST', 'FashionMNIST'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--noise-std', type=float, default=0.0)
    parser.add_argument('--anneal-alpha-init', type=float, default=0.142857)
    parser.add_argument('--anneal-gamma', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
    parser.add_argument('--run-remark', type=str, default="")

    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    EPOCH = args.epochs
    anneal_alpha = args.anneal_alpha_init

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    eval_batch_size = 256
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    lsoftmax = torch.nn.LogSoftmax(dim=1)
    train_kwargs = {'drop_last':True}  # drop last True?
    kwargs = {'drop_last':False}  # drop last True?


    config_dict = {
    "in_dist": args.in_dist_dataset,
    "learning_rate": lr,
    "epochs": EPOCH,
    "batch_size": batch_size,
    "eval_batch_size": eval_batch_size,
    "model_name": args.model,
    "weight_decay": args.weight_decay,
    "noise_std": args.noise_std,
    "anneal_alpha_init": args.anneal_alpha_init,
    "anneal_gamma": args.anneal_gamma,
    "run_remark": args.run_remark,
    }

    # anonymous
    # wandb.init(...)

    # Greyscale
    img_channels = 1

    net = eval(args.model)(out_dim=2, img_channels=img_channels)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)



    for e in range(EPOCH):
        if args.in_dist_dataset == "MNIST" or args.in_dist_dataset == "FashionMNIST":
            train_loader, train_loader_2, test_loader, oe_dataloader, m_or_f_test_dl, kmnist_test_dl, notmnist_test_dl, omni_test_dl, crop_resize_dl = greyscale_dataloader.get_dataloaders(args.in_dist_dataset, batch_size, eval_batch_size, args.data_dir, train_kwargs, kwargs)
        else:
            raise NotImplementedError

        net.train()
        train_loss = 0.0
        train_N = 0
        for (x_in, _), (x_in_2, _), (x_out, _) in zip(train_loader, train_loader_2, crop_resize_dl):
            optimizer.zero_grad()
            x_in    = x_in.to(device)
            x_in_2  = x_in_2.to(device)
            x_out   = x_out.to(device)

            assert x_in.shape[0] == x_out.shape[0]
            # annealing
            anneal_mask = torch.bernoulli(torch.ones(x_in.shape[0], device=device) * anneal_alpha).unsqueeze(1).repeat(1, img_channels*28*28).view(x_in.shape[0], img_channels, 28, 28)
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
        m_or_f_scores = get_scores(m_or_f_test_dl)[:10000]
        kmnist_scores = get_scores(kmnist_test_dl)[:10000]
        notmnist_scores = get_scores(notmnist_test_dl)[:10000]
        omni_scores = get_scores(omni_test_dl)[:10000]

        m_or_f_scores_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, m_or_f_scores)))
        kmnist_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, kmnist_scores)))
        notmnist_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, notmnist_scores)))
        omni_AUROC = roc_auc_score([0]*10000 + [1] * 10000, np.concatenate((test_scores, omni_scores)))

        m_or_f_scores_AUROC_name = "FashionMNIST_scores_AUROC" if args.in_dist_dataset == "MNIST" else "MNIST_scores_AUROC"
        print(f"epoch {e}: train_loss: {train_loss / train_N:.4f}, {m_or_f_scores_AUROC_name}: {m_or_f_scores_AUROC: .4f}, kmnist_AUROC: {kmnist_AUROC:.4f}, notmnist_AUROC: {notmnist_AUROC:.4f}, omni_AUROC: {omni_AUROC:.4f}")

        if args.in_dist_dataset == "MNIST":
            wandb.log({
                "FashionMNIST_scores_AUROC": m_or_f_scores_AUROC
                }, step=e)
        else:
            wandb.log({
                "MNIST_scores_AUROC": m_or_f_scores_AUROC
                }, step=e)

        wandb.log({"train_loss": train_loss / train_N,
                "kmnist_AUROC": kmnist_AUROC,
                "notmnist_AUROC": notmnist_AUROC,
                "omni_AUROC": omni_AUROC
                }, step=e)

    wandb.finish()

