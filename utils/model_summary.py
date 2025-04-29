from torchinfo import summary
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D
# Assuming the model classes are defined above or imported

def print_model_summaries():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chnum_in = 1  # or 3 for RGB input
    dummy_input = (1, chnum_in, 16, 128, 128)  # (batch, channel, depth, height, width)

    print("="*60)
    print("AutoEncoderCov3D Summary")
    print("="*60)
    ae_model = AutoEncoderCov3D(chnum_in).to(device)
    summary(ae_model, input_size=dummy_input, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

    print("\n" + "="*60)
    print("GatedAutoEncoderCov3D Summary")
    print("="*60)
    gated_model = GatedAutoEncoderCov3D(chnum_in).to(device)
    summary(gated_model, input_size=dummy_input, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

if __name__ == "__main__":
    print_model_summaries()
