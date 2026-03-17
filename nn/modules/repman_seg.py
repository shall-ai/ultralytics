import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import RepConv, DWConv


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))




class RepLightConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.rep = RepConv(c2, c2, 3, 1)

    def forward(self, x):
        return self.rep(self.cv1(x))


class RepHGBlock(nn.Module):
    def __init__(self, c1, c2, n=4, shortcut=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_mid = c2 // 2
        self.stem = Conv(c1, c_mid, 1, 1)
        for _ in range(n):
            self.blocks.append(RepLightConv(c_mid, c_mid))
        self.fuse = Conv(c_mid * (n + 1), c2, 1, 1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        y = []
        x1 = self.stem(x)
        y.append(x1)
        z = x1
        for m in self.blocks:
            z = m(z)
            y.append(z)
        out = self.fuse(torch.cat(y, 1))
        return out + x if self.shortcut else out


class Stem(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 2, 1, p=0)
        self.cv3 = Conv(c_ * 2, c2, 2, 1, p=0)   # 这里改成 2*c_
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        x1 = self.cv2(F.pad(x, (0, 1, 0, 1)))
        x2 = self.pool(F.pad(x, (0, 1, 0, 1)))
        x = torch.cat([x1, x2], 1)   # 通道翻倍
        x = self.cv3(x)
        return x


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class MANet(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        c_ = c2 // 2
        self.n = n

        self.cv_in = Conv(c1, 2 * c_, 1, 1)
        self.branch1 = Conv(2 * c_, c_, 1, 1)
        self.branch2 = nn.Sequential(
            Conv(2 * c_, 2 * c_, 1, 1),
            DWConv(2 * c_, c_, 3, 1)
        )

        self.split_conv = Conv(2 * c_, 2 * c_, 1, 1)
        self.blocks = nn.ModuleList([Bottleneck(c_, c_) for _ in range(n)])

        self.cv_out = Conv((3 + n) * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv_in(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)

        x1, x2 = self.split_conv(x).chunk(2, 1)

        feats = [b1, b2, x1]
        z = x2
        for m in self.blocks:
            z = m(z)
            feats.append(z)

        x = torch.cat(feats, 1)
        return self.cv_out(x)