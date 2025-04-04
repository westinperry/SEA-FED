import torch
from torch import nn
import torch.nn.functional as F

# -------------------------------
# Simple AutoEncoder (AE)
# -------------------------------
class AutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super(AutoEncoderCov3D, self).__init__()
        # (Copy the simpler encoder-decoder structure from your simpler file)
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256

        # Encoder
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv3d(chnum_in, feature_num_2, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(feature_num_2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(feature_num_2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(feature_num),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(feature_num, feature_num_x2, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(feature_num_x2),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(feature_num_x2, feature_num_x2, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(feature_num_x2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3,3,3), stride=(2,2,2), 
                               padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3,3,3), stride=(2,2,2), 
                               padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3,3,3), stride=(2,2,2), 
                               padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, chnum_in, (3,3,3), stride=(1,2,2), 
                               padding=(1,1,1), output_padding=(0,1,1))
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out

# -------------------------------
# Gated AutoEncoder (Gated_AE)
# -------------------------------
class GatedAutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super(GatedAutoEncoderCov3D, self).__init__()
        self.chnum_in = chnum_in
        # Define feature sizes
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256

        # Shared Encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv3d(chnum_in, feature_num_2, kernel_size=3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Private Encoder
        self.private_encoder = nn.Sequential(
            nn.Conv3d(feature_num_2, feature_num, kernel_size=3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, kernel_size=3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, kernel_size=3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Global Decoder (with skip connection)
        self.global_decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2 + feature_num_2, feature_num_x2, kernel_size=3, stride=(2,2,2), 
                               padding=1, output_padding=1),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, kernel_size=3, stride=(2,2,2), 
                               padding=1, output_padding=1),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, kernel_size=3, stride=(2,2,2), 
                               padding=1, output_padding=1),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, chnum_in, kernel_size=3, stride=(1,2,2), 
                               padding=1, output_padding=(0,1,1))
        )

        # Local Decoder (with skip connection)
        self.local_decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2 + feature_num_2, feature_num_x2, kernel_size=3, stride=(2,2,2), 
                               padding=1, output_padding=1),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, kernel_size=3, stride=(2,2,2), 
                               padding=1, output_padding=1),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, kernel_size=3, stride=(2,2,2), 
                               padding=1, output_padding=1),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, chnum_in, kernel_size=3, stride=(1,2,2), 
                               padding=1, output_padding=(0,1,1))
        )

        # Fusion Module for adaptive mixing
        self.global_fc = nn.Sequential(
            nn.Linear(feature_num_x2, feature_num_x2 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_num_x2 // 4, chnum_in),
            nn.Sigmoid()
        )
        self.local_fc = nn.Sequential(
            nn.Linear(feature_num_x2, feature_num_x2 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_num_x2 // 4, chnum_in),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_features = self.shared_encoder(x)
        private_features = self.private_encoder(shared_features)
        # Upsample shared features to match private features
        if shared_features.shape[2:] != private_features.shape[2:]:
            shared_resized = F.interpolate(shared_features, size=private_features.shape[2:], mode='trilinear', align_corners=False)
        else:
            shared_resized = shared_features
        # Concatenate for skip connections
        concat_features = torch.cat([private_features, shared_resized], dim=1)
        # Decode using both decoders
        global_out = self.global_decoder(concat_features)
        local_out = self.local_decoder(concat_features)
        # Compute fusion weights
        gap = F.adaptive_avg_pool3d(private_features, output_size=1).view(private_features.size(0), -1)
        global_weight = self.global_fc(gap).view(private_features.size(0), self.chnum_in, 1, 1, 1)
        local_weight = self.local_fc(gap).view(private_features.size(0), self.chnum_in, 1, 1, 1)
        weight_sum = global_weight + local_weight + 1e-8
        global_weight = global_weight / weight_sum
        local_weight = local_weight / weight_sum
        out = global_weight * global_out + local_weight * local_out
        return out

