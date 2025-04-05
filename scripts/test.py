import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D

def load_frames(file_path):
    """
    Loads frames from a file.
    Supports multi-page TIFFs and single-frame JPEG/PNG images.
    Returns a numpy array with shape (T, H, W) where T is the number of frames.
    """
    file_lower = file_path.lower()
    if file_lower.endswith(('.tif', '.tiff')):
        frames_np = tifffile.imread(file_path)
        if frames_np.ndim == 2:
            frames_np = np.expand_dims(frames_np, axis=0)
        return frames_np
    elif file_lower.endswith(('.jpg', '.jpeg', '.png')):
        img = Image.open(file_path).convert('L')
        frames_np = np.array(img)
        if frames_np.ndim == 2:
            frames_np = np.expand_dims(frames_np, axis=0)
        return frames_np
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main(args):
    # Set device and seed
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    utils.seed(args.seed)
    if args.is_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Instantiate the model
    if args.model_name == 'AE':
        model = AutoEncoderCov3D(args.img_channels)
    elif args.model_name == 'Gated_AE':
        model = GatedAutoEncoderCov3D(args.img_channels)
    else:
        raise ValueError("Unknown model name provided.")

    # Load model weights from the checkpoint
    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"Loading model weights from: {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
    else:
        raise ValueError("A valid resume path must be provided.")

    model.to(device)
    model.eval()

    # Loss function (MSE) for anomaly scoring
    loss_func = nn.MSELoss().to(device)

    # Define transforms (same as during training)
    frame_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # List supported files in the evaluation folder (sorted alphabetically)
    supported_exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
    eval_files = [os.path.join(args.eval_folder, f)
                  for f in os.listdir(args.eval_folder) if f.lower().endswith(supported_exts)]
    eval_files = sorted(eval_files)

    if len(eval_files) == 0:
        print("No supported image files found in the provided folder.")
        return

    anomaly_results = []

    # Process each file as one anomaly candidate (using its first frame)
    for file in eval_files:
        print(f"\nProcessing file: {file}")
        try:
            frames_np = load_frames(file)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

        # Use the first frame for anomaly detection.
        first_frame = frames_np[0]
        try:
            tensor_frame = frame_trans(first_frame)
        except Exception as e:
            print(f"Error applying transform on {file}: {e}")
            continue

        # Reshape to (batch, channels, temporal, height, width)
        tensor_frame = tensor_frame.unsqueeze(0).unsqueeze(2)  # shape: [1, C, 1, H, W]
        # Replicate the frame along the temporal dimension to match expected size
        tensor_frame = tensor_frame.repeat(1, 1, args.temporal_frames, 1, 1)  # shape: [1, C, T, H, W]
        tensor_frame = tensor_frame.to(device)

        # Run inference
        with torch.no_grad():
            reconstruction = model(tensor_frame)
            anomaly_score = loss_func(reconstruction, tensor_frame).item()

        # Extract one representative frame for visualization (using the first temporal frame)
        input_frame = tensor_frame[0, 0, 0].cpu().numpy()
        recon_frame = reconstruction[0, 0, 0].cpu().numpy()
        diff_frame = np.abs(recon_frame - input_frame)

        anomaly_results.append({
            "file": file,
            "score": anomaly_score,
            "input": input_frame,
            "recon": recon_frame,
            "diff": diff_frame
        })
        print(f"Raw anomaly score for {os.path.basename(file)}: {anomaly_score:.6f}")

    # Normalize anomaly scores across all files (min-max normalization)
    if len(anomaly_results) > 0:
        raw_scores = np.array([res["score"] for res in anomaly_results])
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score - min_score > 0:
            for res in anomaly_results:
                res["norm_score"] = (res["score"] - min_score) / (max_score - min_score)
        else:
            for res in anomaly_results:
                res["norm_score"] = res["score"]
    else:
        print("No anomalies detected.")
        return

    # Determine which anomaly indices to display.
    # The --frames argument now refers to the global anomaly index (order in the sorted file list)
    if args.frames:
        try:
            indices = list(map(int, args.frames.split(',')))
        except Exception as e:
            print(f"Error parsing anomaly indices: {e}")
            return
    else:
        # If not specified, display all anomalies.
        indices = list(range(len(anomaly_results)))

    for idx in indices:
        if idx < 0 or idx >= len(anomaly_results):
            print(f"Anomaly index {idx} is out of range (total anomalies: {len(anomaly_results)}).")
            continue

        result = anomaly_results[idx]
        print(f"\nDisplaying anomaly index {idx} for file: {result['file']}")
        print(f"Normalized Anomaly Score: {result['norm_score']:.4f}")

        if args.plot:
            plt.figure()
            plt.imshow(result['input'], cmap='gray')
            plt.title(f"Input - {os.path.basename(result['file'])} (Anomaly {idx})")
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.imshow(result['recon'], cmap='gray')
            plt.title(f"Reconstruction - {os.path.basename(result['file'])} (Anomaly {idx})")
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.imshow(result['diff'], cmap='hot')
            plt.title(f"Difference Map - {os.path.basename(result['file'])} (Anomaly {idx})")
            plt.colorbar()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for anomaly detection using an Autoencoder. "
                    "Each file is treated as one anomaly candidate (using its first frame, replicated to form a temporal sequence). "
                    "The --frames argument specifies the global anomaly indices to display (e.g., '0,1,2')."
    )
    parser.add_argument("--eval_folder", type=str, required=True,
                        help="Path to folder containing image files for evaluation")
    parser.add_argument("--resume_path", type=str, required=True,
                        help="Path to model checkpoint to load")
    parser.add_argument("--model_name", type=str, choices=['AE', 'Gated_AE'], default='AE',
                        help="Name of the model to use")
    parser.add_argument("--img_channels", type=int, default=1,
                        help="Number of image channels")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--is_deterministic", action="store_true",
                        help="Set cudnn deterministic behavior")
    # The --frames argument now indicates global anomaly indices (e.g., "0,1,2")
    parser.add_argument("--frames", type=str, default=None,
                        help="Comma-separated list of anomaly indices to display (default: display all anomalies)")
    parser.add_argument("--no_plot", dest="plot", action="store_false",
                        help="Disable plotting of results")
    parser.set_defaults(plot=True)
    # Number of temporal frames to replicate for each input (default should match model's expected temporal dim)
    parser.add_argument("--temporal_frames", type=int, default=8,
                        help="Number of temporal frames to replicate for each input (default: 8)")

    args = parser.parse_args()
    main(args)
