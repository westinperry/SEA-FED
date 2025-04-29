import torch
from torch import nn
import torch.nn.functional as F

class AutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super().__init__()
        f2, f1, f3 = 32, 64, 128
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(chnum_in, f2, 3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(f2), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(f2, f1, 3, stride=2, padding=1),
            nn.BatchNorm3d(f1), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(f1, f3, 3, stride=2, padding=1),
            nn.BatchNorm3d(f3), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(f3, f3, 3, stride=2, padding=1),
            nn.BatchNorm3d(f3), nn.LeakyReLU(0.2, inplace=False),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(f3, f3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(f3), nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose3d(f3, f1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(f1), nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose3d(f1, f2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(f2), nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose3d(f2, chnum_in, 3, stride=(1,2,2), padding=1, output_padding=(0,1,1)),
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out

class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w

class Adapter3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv3d(channels, channels // 4, kernel_size=1)
        self.act  = nn.ReLU(inplace=True)
        self.up   = nn.Conv3d(channels // 4, channels, kernel_size=1)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

class GatedAutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super().__init__()
        f2, f1, f3 = 32, 64, 128

        # Encoder + SE + Adapter
        self.enc1 = self._enc_block(chnum_in, f2)
        self.se1 = SEBlock3D(f2)
        self.adapter1 = Adapter3D(f2)

        self.enc2 = self._enc_block(f2, f1)
        self.se2 = SEBlock3D(f1)
        self.adapter2 = Adapter3D(f1)

        self.enc3 = self._enc_block(f1, f3)
        self.se3 = SEBlock3D(f3)
        self.adapter3 = Adapter3D(f3)

        self.enc4 = self._enc_block(f3, f3)
        self.se4 = SEBlock3D(f3)
        self.adapter4 = Adapter3D(f3)

        # Decoder with upsampling + convolution
        self.dec1 = self._dec_block(f3, f3)
        self.dec2 = self._dec_block(f3, f1)
        self.dec3 = self._dec_block(f1, f2)
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(f2, chnum_in, 3, padding=1)
        )

        # Initialize weights
        self._initialize_weights()

    def _enc_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, stride=2, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.LeakyReLU(0.2)
        )

    def _dec_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(ch_in, ch_out, 3, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.LeakyReLU(0.2)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.se1(e1) + self.adapter1(e1)

        e2 = self.enc2(e1)
        e2 = self.se2(e2) + self.adapter2(e2)

        e3 = self.enc3(e2)
        e3 = self.se3(e3) + self.adapter3(e3)

        e4 = self.enc4(e3)
        e4 = self.se4(e4) + self.adapter4(e4)

        # Decoder with skip connections
        d1 = self.dec1(e4) + e3  # Skip connection from e3
        d2 = self.dec2(d1) + e2   # Skip connection from e2
        d3 = self.dec3(d2) + e1   # Skip connection from e1
        x = self.dec4(d3)

        return x


