import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class ProbAIMnistDataset(Dataset):
    def __init__(self, mnist_dataset, crop_noise=False, crop_size=12, shuffle_pairs=False):
        self.mnist_dataset = mnist_dataset
        self.crop_noise = crop_noise
        self.shuffle_pairs = shuffle_pairs
        if shuffle_pairs:
            self.shuffle_idx = torch.randperm(len(mnist_dataset))
            
        # Can remain unchanged, but feel free to experiment with this parameter
        self.crop_size = crop_size
        
        self.img_dim0 = mnist_dataset.data.shape[1]
        self.img_dim1 = mnist_dataset.data.shape[2]

    def __len__(self):
        return self.mnist_dataset.__len__()

    def __getitem__(self, idx):
        # We don't care about the label so we only extact [0] from the wrapped dataset
        x1 = self.mnist_dataset.__getitem__(idx)[0].detach()
        if self.crop_noise:
            if self.shuffle_pairs:
                idx = self.shuffle_idx[idx]
            x0 = self.mnist_dataset.__getitem__(idx)[0].detach().clone()

            # Sample a square of pixels and set them to 1 (white)
            idx0 = torch.randint(0, self.img_dim0-self.crop_size, (1,))[0]
            idx1 = torch.randint(0, self.img_dim1-self.crop_size, (1,))[0]
            x0[0, idx0:idx0+self.crop_size, idx1:idx1+self.crop_size] = 1
        else:
            x0 = torch.randn(1, self.img_dim0, self.img_dim0)
        return x0,x1

    def reshuffle_pairs(self):
        # You can call this between epochs to rearange the pairs
        # Is probably not that influential/important given the size of the dataset and how few epochs we need
        if self.shuffle_pairs:
            self.shuffle_idx = torch.randperm(len(self.mnist_dataset))


# Load MNIST datsets, init ProbAI dataset objects and create train and test dataloaders
batch_size = YOUR_BATCH_SIZE
crop = YOUR_CROP_SETTING       # True if config 2 and 3
shuffle = YOUR_SHUFFLE_SETTING # True if config 2
trainset = datasets.MNIST(
    "YOUR/DATA/PATH",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
)

testset = datasets.MNIST(
    "YOUR/DATA/PATH",
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
)


train_loader = torch.utils.data.DataLoader(
    ProbAIMnistDataset(trainset, crop_noise=crop, shuffle_pairs=shuffle), batch_size=batch_size, shuffle=True, drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    ProbAIMnistDataset(testset, crop_noise=crop, shuffle_pairs=shuffle), batch_size=batch_size, drop_last=True
)