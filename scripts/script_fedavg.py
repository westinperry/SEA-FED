import torch
import copy
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D

def load_model_weights(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return state_dict

def split_components(state_dict):
    # Split components based on the new naming.
    shared_encoder = {k: v for k, v in state_dict.items() if k.startswith('shared_encoder')}
    private_encoder = {k: v for k, v in state_dict.items() if k.startswith('private_encoder')}
    global_decoder = {k: v for k, v in state_dict.items() if k.startswith('global_decoder')}
    local_decoder = {k: v for k, v in state_dict.items() if k.startswith('local_decoder')}
    # Fusion module parameters for adaptive weighting.
    fusion = {k: v for k, v in state_dict.items() if k.startswith('global_fc') or k.startswith('local_fc')}
    return shared_encoder, private_encoder, global_decoder, local_decoder, fusion

def fed_avg(dicts, weights=None):
    if weights is None:
        weights = [1.0 / len(dicts)] * len(dicts)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    avg_dict = copy.deepcopy(dicts[0])
    for key in avg_dict:
        avg_dict[key] = torch.zeros_like(avg_dict[key], dtype=torch.float)

    for d, w in zip(dicts, weights):
        for key in d:
            avg_dict[key] += w * d[key].to(dtype=torch.float)
    return avg_dict

def merge_components(shared_encoder, private_encoder, global_decoder, local_decoder, fusion):
    merged = {}
    merged.update(shared_encoder)
    merged.update(private_encoder)
    merged.update(global_decoder)
    merged.update(local_decoder)
    merged.update(fusion)
    return merged

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = args.input_paths
    save_paths = args.output_paths

    if len(model_paths) != len(save_paths):
        raise ValueError("Number of input paths must match number of output paths")
    
    # Load state dictionaries
    state_dicts = [load_model_weights(path, device) for path in model_paths]

    if args.ModelName == 'AE':
        # Use simple averaging from first version
        avg_state_dict = fed_avg(state_dicts)
        for save_path in save_paths:
            torch.save(avg_state_dict, save_path)
            print(f"✅ Averaged model saved to {save_path}")
    elif args.ModelName == 'Gated_AE':
        # Use component-wise averaging from the second version
        splits = [split_components(sd) for sd in state_dicts]
        shared_encoders = [s for s, _, _, _, _ in splits]
        private_encoders = [p for _, p, _, _, _ in splits]
        global_decoders = [g for _, _, g, _, _ in splits]
        local_decoders = [l for _, _, _, l, _ in splits]
        fusions = [f for _, _, _, _, f in splits]

        avg_shared_encoder = fed_avg(shared_encoders)
        avg_global_decoder = fed_avg(global_decoders)
        avg_fusion = fed_avg(fusions)

        for i in range(len(model_paths)):
            updated_state_dict = merge_components(
                avg_shared_encoder,
                private_encoders[i],
                avg_global_decoder,
                local_decoders[i],
                avg_fusion
            )
            # You may need to pass the correct channel number here.
            updated_model = GatedAutoEncoderCov3D(chnum_in=1).to(device)
            updated_model.load_state_dict(updated_state_dict)
            torch.save(updated_model.state_dict(), save_paths[i])
            print(f"✅ Updated model {i+1} saved to {save_paths[i]}")
    else:
        raise ValueError("Unknown ModelName")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Model Averaging")
    parser.add_argument('--input-paths', nargs='+', required=True,
                        help='List of input model paths')
    parser.add_argument('--output-paths', nargs='+', required=True,
                        help='List of output model paths')
    parser.add_argument('--ModelName', type=str, required=True,
                        help='Model name: AE (simple) or Gated_AE (with gates)')
    
    args = parser.parse_args()
    main(args)
