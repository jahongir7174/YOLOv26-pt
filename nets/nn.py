import copy
import math

import torch

from utils.util import make_anchors


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, c1, c2, relu, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, in_ch, out_ch, add=True, e=0.5):
        super().__init__()
        self.add_m = add and in_ch == out_ch
        self.conv1 = Conv(in_ch, int(out_ch * e), torch.nn.SiLU(), k=3, s=1, p=1)
        self.conv2 = Conv(int(out_ch * e), out_ch, torch.nn.SiLU(), k=3, s=1, p=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add_m else y


class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        self.qkv = Conv(dim, dim + (self.key_dim * num_heads) * 2, torch.nn.Identity())
        self.proj = Conv(dim, dim, torch.nn.Identity())
        self.pe = Conv(dim, dim, torch.nn.Identity(), 3, 1, 1, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, h * w)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        x = self.proj(x)
        return x


class PSABlock(torch.nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, add=True):
        super().__init__()
        self.add_m = add
        self.conv1 = Attention(c, num_heads, attn_ratio)
        self.conv2 = torch.nn.Sequential(Conv(c, c * 2, torch.nn.SiLU()),
                                         Conv(c * 2, c, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x) if self.add_m else self.conv1(x)
        x = x + self.conv2(x) if self.add_m else self.conv2(x)
        return x


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch, add=True, e=0.5):
        super().__init__()
        self.conv1 = Conv(in_ch, int(out_ch * e), torch.nn.SiLU())
        self.conv2 = Conv(in_ch, int(out_ch * e), torch.nn.SiLU())
        self.conv3 = Conv(2 * int(out_ch * e), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(int(out_ch * e), int(out_ch * e), add, e=1.0),
                                         Residual(int(out_ch * e), int(out_ch * e), add, e=1.0))

    def forward(self, x):
        return self.conv3(torch.cat(tensors=(self.res_m(self.conv1(x)), self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, c1, c2, n=1, csp=False, e=0.5, attn=False, add=True):
        super().__init__()
        self.conv1 = Conv(c1, 2 * int(c2 * e), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * int(c2 * e), c2, torch.nn.SiLU())

        modules = []
        for _ in range(n):
            if csp:
                if attn:
                    modules.append(torch.nn.Sequential(Residual(int(c2 * e), int(c2 * e), add),
                                                       PSABlock(int(c2 * e), num_heads=max(int(c2 * e) // 64, 1))))
                else:
                    modules.append(CSPModule(int(c2 * e), int(c2 * e), add))
            else:
                modules.append(Residual(int(c2 * e), int(c2 * e), add))
        self.res_m = torch.nn.ModuleList(modules)

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, n=3, add=True):
        super().__init__()
        self.n = n
        self.add_m = add and in_ch == out_ch
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.Identity())
        self.conv2 = Conv((in_ch // 2) * (n + 1), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.conv1(x)]
        y.extend(self.res_m(y[-1]) for _ in range(self.n))
        y = self.conv2(torch.cat(y, dim=1))
        return x + y if self.add_m else y


class PSA(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, e=0.5):
        super().__init__()
        assert in_ch == out_ch
        self.c = int(in_ch * e)
        self.conv1 = Conv(in_ch, 2 * self.c, torch.nn.SiLU())
        self.conv2 = Conv(2 * self.c, in_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(self.c, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.conv1(x).split((self.c, self.c), dim=1)
        return self.conv2(torch.cat(tensors=(a, self.res_m(b)), dim=1))


class Backbone(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], e=0.25))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], e=0.25))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1]))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1]))
        self.p5.append(SPP(width[5], width[5]))
        self.p5.append(PSA(width[5], width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]


class Neck(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp=csp[1])
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp=csp[1])
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp=csp[1])
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], csp=csp[1], attn=True)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return [p3, p4, p5]


class Head(torch.nn.Module):
    shape = None
    export = False
    max_det = 300
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc, filters):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 4  # number of outputs per anchor
        self.nl = len(filters)  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max((16, filters[0] // 4))
        cls = max(filters[0], max(min(self.nc, 100), 80))

        self.box_train = torch.nn.ModuleList(torch.nn.Sequential(Conv(i, box, torch.nn.SiLU(), 3, p=1),
                                                                 Conv(box, box, torch.nn.SiLU(), 3, p=1),
                                                                 torch.nn.Conv2d(box,
                                                                                 out_channels=4, kernel_size=1))
                                             for i in filters)
        self.cls_train = torch.nn.ModuleList(torch.nn.Sequential(Conv(i, i, torch.nn.SiLU(), k=3, p=1, g=i),
                                                                 Conv(i, cls, torch.nn.SiLU()),
                                                                 Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                                 Conv(cls, cls, torch.nn.SiLU()),
                                                                 torch.nn.Conv2d(cls,
                                                                                 out_channels=self.nc, kernel_size=1))
                                             for i in filters)

        self.box_head = copy.deepcopy(self.box_train)
        self.cls_head = copy.deepcopy(self.cls_train)

    def __forward(self, x, box_head, cls_head):
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(x=x, boxes=boxes, scores=scores)

    def forward(self, x):
        if self.training:
            y1 = self.__forward(x, self.box_train, self.cls_train)
            y2 = self.__forward([i.detach() for i in x], self.box_head, self.cls_head)
            return y1, y2
        else:
            x = [i.detach() for i in x]
            y = self.__forward(x, self.box_head, self.cls_head)
            shape = y["x"][0].shape
            if self.shape != shape:
                self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(y["x"], self.stride))
                self.shape = shape

            box = y['boxes']
            anchors = self.anchors.unsqueeze(0)
            lt, rb = box.chunk(2, 1)
            box = torch.cat((anchors - lt, anchors + rb), 1) * self.strides
            y = torch.cat((box, y["scores"].sigmoid()), dim=1).permute(0, 2, 1)
            boxes, scores = y.split([4, self.nc], dim=-1)
            batch_size, anchors, nc = scores.shape
            k = self.max_det if self.export else min(self.max_det, anchors)
            ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
            scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
            scores, index = scores.flatten(1).topk(k)
            idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
            scores = scores[..., None]
            conf = (index % nc)[..., None].float()
            boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
            return torch.cat([boxes, scores, conf], dim=-1)

    def initialize_biases(self):
        """
        Initialize biases, WARNING: requires stride availability.
        """
        for i, (a, b) in enumerate(zip(self.box_train, self.cls_train)):
            # box
            a[-1].bias.data[:] = 2.0
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        for i, (a, b) in enumerate(zip(self.box_head, self.cls_head)):
            # box
            a[-1].bias.data[:] = 2.0
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.backbone = Backbone(width, depth, csp)
        self.neck = Neck(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))

        outputs = self.forward(img_dummy)[0]['x']
        self.head.stride = torch.tensor([256 / i.shape[-2] for i in outputs])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class Detector(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.backbone = Backbone(width, depth, csp)
        self.neck = Neck(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))

        outputs = self.forward(img_dummy)[0]['x']
        self.head.stride = torch.tensor([256 / i.shape[-2] for i in outputs])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v26_n(num_classes: int):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return Detector(width, depth, csp, num_classes)


def yolo_v26_t(num_classes: int):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, csp, num_classes)


def yolo_v26_s(num_classes: int):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return Detector(width, depth, csp, num_classes)


def yolo_v26_m(num_classes: int):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v26_l(num_classes: int):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v26_x(num_classes: int):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes)
