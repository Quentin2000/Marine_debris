
from torch import nn
import torchvision.transforms as T
import torch
import random

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class RandomRot90(nn.Module):
    def __init__(self, dims=[-2, -1]):
        super().__init__()
        self.dims = dims

    def forward(self,x):
        rot = torch.randint(high=4,size=(1,))
        return torch.rot90(x, int(rot), self.dims)

class PixelNoise(nn.Module):
    """
    for each pixel, same across all bands
    """
    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x):
        C, H, W = x.shape
        noise_level = x.std() * self.std_noise
        pixel_noise = torch.rand(H, W, device=x.device)
        return x + pixel_noise.view(1,H,W) * noise_level

class ChannelNoise(nn.Module):
    """
    for each channel
    """
    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x):
        C, H, W = x.shape
        noise_level = x.std() * self.std_noise

        channel_noise = torch.rand(C, device=x.device)
        return x + channel_noise.view(-1,1,1).to(x.device) * noise_level

class Noise(nn.Module):
    """
    for each channel
    """
    def __init__(self, std_noise=0.1):
        super().__init__()
        self.std_noise = std_noise

    def forward(self, x):
        noise_level = x.std() * self.std_noise
        noise = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=x.device)
        return x + noise * noise_level


def get_train_transform(crop_size=64):
    return torch.nn.Sequential(
            #RandomApply(
            #    T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            #    p = 0.3
            #),
            #T.RandomGrayscale(p=0.2),
            RandomRot90(),
            #T.RandomRotation(90),
            T.RandomResizedCrop(crop_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)),p = 0.6),
            RandomApply(PixelNoise(std_noise=0.25),p = 0.6),
            RandomApply(ChannelNoise(std_noise=0.25),p = 0.6),
            RandomApply(Noise(std_noise=0.25),p = 0.6),
        )
