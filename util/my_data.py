from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader


def default_loader(path):
    return Image.open(path)


class Froth(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        super(Froth, self).__init__()

        file = open(txt, 'r')
        images = []
        for line in file:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格
            images.append((words[0], int(words[1])))  # 把txt里的内容读入img列表保存，具体是words几要看txt内容而定

        self.image = images
        self.loader = loader
        self.transform = transform

    def __getitem__(self, item):
        fn, label = self.image[item]
        img = self.loader(fn)  # 从地址中读取图片

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image)


def FashionMNIST(train_size, test_size, num_workers=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    train_data = torchvision.datasets.FashionMNIST(root='../data', train=True, download=False, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=train_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader


def cifar_mini(train_size, test_size, num_workers=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=False, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=train_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader


def cifar(train_size, test_size, num_workers=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR100(root='../data/cifar-100', train=True, download=False, transform=transform)
    test_data = torchvision.datasets.CIFAR100(root='../data/cifar-100', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=train_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader
