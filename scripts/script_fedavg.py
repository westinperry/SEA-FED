#!/usr/bin/env python3
import torch
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D  # Ensure this imports the improved model

def load_model_weights(path, device):
    state = torch.load(path, map_location=device)
    return {k.replace('module.', ''): v for k, v in state.get('state_dict', state).items()}

def fed_avg(dicts, weights=None, skip_keys=None):
    n = len(dicts)
    weights = [1.0/n]*n if weights is None else [w/sum(weights) for w in weights]
    
    avg = {}
    for key in dicts[0]:
        if skip_keys and any(sk in key for sk in skip_keys):
            continue
        avg[key] = sum(d[key].float() * w for d, w in zip(dicts, weights)).type_as(dicts[0][key])
    return avg

def main():
    parser = argparse.ArgumentParser(description="FedAvg for 3D AE Models")
    parser.add_argument('--input-paths',  nargs='+', required=True,
                      help='Client model paths')
    parser.add_argument('--output-paths', nargs='+', required=True,
                      help='Output paths for personalized models')
    parser.add_argument('--ModelName',    choices=['AE','Gated_AE'], required=True,
                      help='Model architecture')
    parser.add_argument('--Channels',     type=int, default=1,
                      help='Input channels')
    args = parser.parse_args()

    assert len(args.input_paths) == len(args.output_paths), "Input/Output count mismatch!"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dicts = [load_model_weights(p, device) for p in args.input_paths]

    # Aggregation strategy
    skip_keys = ['adapter'] if args.ModelName == 'Gated_AE' else None
    global_avg = fed_avg(state_dicts, skip_keys=skip_keys)

    for i, out_path in enumerate(args.output_paths):
        # Personalization logic
        personalized_sd = global_avg.copy()
        if args.ModelName == 'Gated_AE':
            personalized_sd.update({
                k: v for k, v in state_dicts[i].items()
                if any(sk in k for sk in ['adapter', 'bn'])  # Keep BN layers local if needed
            })
        
        # Initialize with proper weights
        model = (GatedAutoEncoderCov3D if args.ModelName == 'Gated_AE' else AutoEncoderCov3D)(
            args.Channels
        ).to(device)
        model.load_state_dict(personalized_sd, strict=False)
        
        torch.save(model.state_dict(), out_path)
        print(f"✅ Saved personalized model to {out_path}")

if __name__ == "__main__":
    main()