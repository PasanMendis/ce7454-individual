import torch, torch.nn as nn

class DWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        Drop = (lambda p: nn.Identity()) if p <= 0 else (lambda p: nn.Dropout2d(p))
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True), Drop(p),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), Drop(p),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), Drop(p),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), Drop(p),
        )
    def forward(self, x): return self.f(x)

class LiteUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=19, base=64, dropout_p=0.0):
        super().__init__()
        C = [in_ch, base, base*2, base*4, base*8]
        self.e1 = DWBlock(C[0], C[1], dropout_p); self.p1 = nn.MaxPool2d(2)
        self.e2 = DWBlock(C[1], C[2], dropout_p); self.p2 = nn.MaxPool2d(2)
        self.e3 = DWBlock(C[2], C[3], dropout_p); self.p3 = nn.MaxPool2d(2)
        self.b  = DWBlock(C[3], C[4], dropout_p)

        self.u3 = nn.ConvTranspose2d(C[4], C[3], 2, 2); self.d3 = DWBlock(C[3]+C[3], C[3], dropout_p)
        self.u2 = nn.ConvTranspose2d(C[3], C[2], 2, 2); self.d2 = DWBlock(C[2]+C[2], C[2], dropout_p)
        self.u1 = nn.ConvTranspose2d(C[2], C[1], 2, 2); self.d1 = DWBlock(C[1]+C[1], C[1], dropout_p)
        self.head = nn.Conv2d(C[1], num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x); x = self.p1(e1)
        e2 = self.e2(x); x = self.p2(e2)
        e3 = self.e3(x); x = self.p3(e3)
        x  = self.b(x)

        x = self.u3(x); x = torch.cat([x,e3],1); x = self.d3(x)
        x = self.u2(x); x = torch.cat([x,e2],1); x = self.d2(x)
        x = self.u1(x); x = torch.cat([x,e1],1); x = self.d1(x)
        return self.head(x)