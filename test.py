from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch import nn
from torch import optim
from models import resnet_cifar
import models.resnet_cifar as resnet_cifar 
import cifar_train

# CIFAR 10
imb_factor = 0.01
gpu =  0 
imb_type ="exp" 
imb_factor =0.01 
loss_type = "CE" 
train_rule =None
rand_number = 0


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# K means clustering
def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c


if __name__ == '__main__':

        # CIFAR 10
    imb_factor = 0.01
    gpu =  0 
    imb_type ="exp" 
    imb_factor =0.01 
    loss_type = "CE" 
    train_rule =None
    rand_number = 0


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    train_dataset = IMBALANCECIFAR10(root='./data', imb_type=imb_type, imb_factor=imb_factor, rand_number=rand_number, train=True, download=False, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_val)

    # data loader
    num_workers = 4
    batch_size = 128
    train_sampler = None 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)


    # define model
    model = resnet_cifar.resnet20()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 50

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_freq", default = 1)
    parser.add_argument("--gpu", default = None)
    args = parser.parse_args()

    # train model
    cifar_train.train(train_loader, model, criterion, optimizer, epochs, args, open('logfile', 'w'), None)

    # get layers' output
    model.eval()
    output = []
    for i, (inp, target) in enumerate(val_loader):
        output.append(model(inp))
    print(output)
    print("dafadfafdsadf")

    # clustering

    _, clusters = KMeans(np.array(output), K=15, iter=20)
    print(clusters)