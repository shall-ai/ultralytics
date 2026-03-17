import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Conv-BN-Act."""
    default_act = nn.SiLU(inplace=False)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable conv."""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dconv = Conv(c1, c1, k, s, g=c1, act=act)
        self.pconv = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class LSBlock(nn.Module):
    """
    Local Spatial Block
    """
    def __init__(self, c):
        super().__init__()
        self.dw = DWConv(c, c, k=3, s=1)
        self.pw1 = Conv(c, c, k=1, s=1)
        self.pw2 = Conv(c, c, k=1, s=1, act=False)

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.pw2(x)
        return x + identity


class SimpleSS2D(nn.Module):
    """
    工程近似版 SS2D：
    BN -> 大核DWConv -> 门控投影 -> PWConv -> residual
    """
    def __init__(self, c):
        super().__init__()
        self.norm = nn.BatchNorm2d(c)
        self.dw_large = nn.Conv2d(
            c, c, kernel_size=7, stride=1, padding=3, groups=c, bias=False
        )
        self.pw_in = nn.Conv2d(c, c * 2, kernel_size=1, bias=False)
        self.pw_out = nn.Conv2d(c, c, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.dw_large(x)
        x = self.pw_in(x)
        a, b = x.chunk(2, dim=1)
        x = F.silu(a, inplace=False) * torch.sigmoid(b)
        x = self.pw_out(x)
        return x + identity


class ResGatedBlock(nn.Module):
    """
    RG Block / gated MLP 风格
    """
    def __init__(self, c, expansion=2.0):
        super().__init__()
        hidden = int(c * expansion)
        self.fc1 = nn.Conv2d(c, hidden * 2, kernel_size=1, bias=False)
        self.dw = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=False
        )
        self.bn = nn.BatchNorm2d(hidden)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        y = self.fc1(x)
        a, b = y.chunk(2, dim=1)
        a = F.silu(a, inplace=False) * torch.sigmoid(b)
        a = self.dw(a)
        a = F.silu(self.bn(a), inplace=False)
        a = self.fc2(a)
        return a + identity


class MoERouter(nn.Module):
    """
    Soft routing
    """
    def __init__(self, c, num_experts=4):
        super().__init__()
        hidden = max(c // 4, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, num_experts)

    def forward(self, x):
        g = self.pool(x).flatten(1)  # [B, C]
        g = F.silu(self.fc1(g), inplace=False)
        g = self.fc2(g)
        return F.softmax(g, dim=1)  # [B, E]


class MoEODSSBlock(nn.Module):
    """
    Ultralytics-compatible:
    __init__(c1, c2, ...)
    输出通道严格为 c2
    """
    def __init__(self, c1, c2, num_experts=4, expansion=2.0, shortcut=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.shortcut = shortcut and (c1 == c2)

        self.pre = Conv(c1, c2, 3, 1)
        self.ls = LSBlock(c2)
        self.ss2d = SimpleSS2D(c2)
        self.router = MoERouter(c2, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [ResGatedBlock(c2, expansion=expansion) for _ in range(num_experts)]
        )

    def forward(self, x):
        identity = x
        x = self.pre(x)
        x = self.ls(x)
        x = self.ss2d(x)

        weights = self.router(x)  # [B, E]
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            yi = expert(x)
            wi = weights[:, i].view(-1, 1, 1, 1)
            out = out + yi * wi

        if self.shortcut:
            out = out + identity
        return out


class CSA(nn.Module):
    """
    Clustering Self-Attention
    Ultralytics-compatible:
    __init__(c1, c2, num_clusters=3, heads=4)
    输出通道严格为 c2
    """
    def __init__(self, c1, c2, num_clusters=3, heads=4):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = num_clusters
        self.heads = heads

        assert c2 % heads == 0, f"c2={c2} must be divisible by heads={heads}"
        self.d = c2 // heads

        self.proj_in = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        self.cluster_proj = nn.Conv2d(c2, num_clusters, 1)
        self.qkv = nn.Conv2d(c2, c2 * 3, 1, bias=False)
        self.proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.scale = self.d ** -0.5

    def forward(self, x):
        x = self.proj_in(x)
        identity = x

        b, c, h, w = x.shape
        n = h * w

        cluster_logits = self.cluster_proj(x)              # [B, K, H, W]
        cluster_prob = torch.softmax(cluster_logits, dim=1)

        qkv = self.qkv(x).reshape(b, 3, self.heads, self.d, n)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # [B, heads, d, n]
        q = q.permute(0, 1, 3, 2)                          # [B, heads, n, d]
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        out = torch.zeros_like(x)
        for i in range(self.k):
            mask = cluster_prob[:, i:i + 1].reshape(b, 1, n, 1)   # [B,1,n,1]
            qi = q * mask
            ki = k * mask
            vi = v * mask

            attn = (qi @ ki.transpose(-2, -1)) * self.scale       # [B, heads, n, n]
            attn = attn.softmax(dim=-1)
            yi = attn @ vi                                         # [B, heads, n, d]
            yi = yi.permute(0, 1, 3, 2).reshape(b, c, h, w)
            out = out + yi

        out = out / self.k
        out = self.proj(out)
        return out + identity