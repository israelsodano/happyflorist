import math
from torch import nn, torch

class Conv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=2, p=1) -> None:
        super().__init__()
        
        self.in_c = in_c
        self.out_c = out_c
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c,  0.001, 0.03),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        return self.layers(x)
    
class Bottleneck(nn.Module):
    def __init__(self, short_c, in_c) -> None:
        super().__init__()
        self.short_c = short_c
        self.layers = nn.Sequential(
            Conv(in_c, in_c, 3, 1, 1),
            Conv(in_c, in_c, 3, 1, 1)
        )
        
    def forward(self, x):
        return self.layers(x) + x if self.short_c else self.layers(x)

class C2F(nn.Module):
    def __init__(self, in_c, out_c, n, short_c) -> None:
        super().__init__()
        
        self.in_c = in_c
        self.out_c = out_c
        
        self.cv1 = Conv(in_c, out_c//2, 1, 1, 0)
        self.cv2 = Conv(in_c, out_c//2, 1, 1, 0)
        self.cv3 = Conv((n + 2)*out_c//2, out_c, 1, 1, 0)
        self.bn_n = nn.ModuleList([Bottleneck(short_c, out_c//2) for _ in range(n)])
    
    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for b in self.bn_n:
            y.append(b(y[-1]))
        return self.cv3(torch.cat(y, dim=1))
    
class SPPF(nn.Module):
    def __init__(self, in_c, k=5, s=1, p=2) -> None:
        super().__init__()
        self.cv1 = Conv(in_c, in_c//2, 1, 1, 0)
        self.maxp = nn.MaxPool2d(k, s, p)
        self.cv2 = Conv(in_c*2, in_c, 1, 1, 0)
    
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.maxp(y1)
        y3 = self.maxp(y2)
        y4 = self.maxp(y3)
        
        y = torch.cat([y1, y2, y3, y4], dim=1)
        return self.cv2(y)
    
class Backbone(nn.Module):
    def __init__(self, depth, width) -> None:
        super().__init__()
        
        in_c = width[0]
        out_c = width[1]
        self.p1 = nn.Sequential(
            Conv(in_c, out_c)
        )
        
        in_c = out_c
        out_c = width[2]
        self.p2 = nn.Sequential(
            Conv(in_c, out_c), 
            C2F(out_c, out_c, depth[0], True)
        )
        
        in_c = out_c
        out_c = width[3]
        self.p3 = nn.Sequential(
            Conv(in_c, out_c), 
            C2F(out_c, out_c, depth[1], True)
        )
        
        in_c = out_c
        out_c = width[4]
        self.p4 = nn.Sequential(
            Conv(in_c, out_c), 
            C2F(out_c, out_c, depth[2], True)
        )
        
        in_c = out_c
        out_c = width[5]
        self.p5 = nn.Sequential(
            Conv(in_c, out_c), 
            C2F(out_c, out_c, depth[3], True),
            SPPF(out_c)
        )
        
    def forward(self, x):
        
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        
        return p3, p4, p5
    
class Predict(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            Conv(in_c, in_c//2, 3, 1, 1),
            Conv(in_c//2, in_c, 3, 1, 1),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):  
        return self.layers(x)
    
class Neck(nn.Module):
    def __init__(self, depth, width) -> None:
        super().__init__()
        self.up = nn.Upsample(None, scale_factor=2)
        self.h1 = C2F(width[4] + width[5], width[4], depth[0], False)
        self.h2 = C2F(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = C2F(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4])
        self.h6 = C2F(width[4] + width[5], width[5], depth[0], False)
        
        
    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        
        return h2, h4, h6
    
class DFL(torch.nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)
    
def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
    
class Head(nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, num_classes, filters=()) -> None:
        super().__init__()
        self.ch = 16  
        self.nc = num_classes
        self.num_classes = num_classes  
        self.nl = len(filters)  
        self.no = 1 + num_classes + self.ch * 4  
        self.stride = torch.zeros(self.nl)  
        
        c1 = max(filters[0], self.num_classes)
        c2 = max((filters[0] // 4, self.ch * 4))
        
        self.dfl = DFL(self.ch)
        
        self.nobj = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3, 1, 1),
                                                           Conv(c1, c1, 3, 1, 1),
                                                           torch.nn.Conv2d(c1, 1, 1)) for x in filters)
        
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c1, 3, 1, 1),
                                                           Conv(c1, c1, 3, 1, 1),
                                                           torch.nn.Conv2d(c1, self.num_classes, 1)) for x in filters)
        
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c2, 3, 1, 1),
                                                           Conv(c2, c2, 3, 1, 1),
                                                           torch.nn.Conv2d(c2, 4 * self.ch, 1)) for x in filters)
    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cls[i](x[i]), 
                              self.nobj[i](x[i]), 
                              self.box[i](x[i])), 1)
        return x
    
    def initialize_biases(self):
        
        
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  
            
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        
class YoloV8(nn.Module):
    
    def __init__(self, depth, width, num_classes) -> None:
        super().__init__()
        
        self.bb = Backbone(depth, width)
        self.nk = Neck(depth, width)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        
        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()
        
    def forward(self, x):
        x = self.bb(x)
        x = self.nk(x)
        
        return self.head(list(x))

def new_yolo_v8_n(num_classes: int):
    depth = [1, 2, 2, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YoloV8(depth, width, num_classes)

def new_yolo_v8_s(num_classes: int):
    depth = [1, 2, 2, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YoloV8(depth, width, num_classes)

def new_yolo_v8_m(num_classes: int):
    depth = [2, 4, 4, 2]
    width = [3, 48, 96, 192, 384, 576]
    return YoloV8(depth, width, num_classes)

def new_yolo_v8_l(num_classes: int):
    depth = [3, 6, 6, 3]
    width = [3, 64, 128, 256, 512, 512]
    return YoloV8(depth, width, num_classes)

def new_yolo_v8_x(num_classes: int):
    depth = [3, 6, 6, 3]
    width = [3, 80, 160, 320, 640, 640]
    return YoloV8(depth, width, num_classes)
    