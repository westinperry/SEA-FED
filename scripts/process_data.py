import os
import numpy as np
import scipy.io
import utils

def preprocess_images():
    """Preprocess images and generate ground truth labels."""
    print("Starting image preprocessing...")
    
    # Paths
    data_root_path = '/home/westin/Documents/Code/'
    in_path = os.path.join(data_root_path, 'datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/')  # Matches visible UCSDped1
    out_path = os.path.join(data_root_path, 'datasets/processed/UCSD_P2_256/')  # Output for UCSDped1
    
    utils.mkdirfunc(out_path)
    
    # Directory and file counts for UCSDped1
    sub_dir_list = ['Train', 'Test']
    file_num_list = [16, 12]  # 34 Train, 36 Test
    
    # Preprocessing options
    opts = {
        'is_gray': True,
        'outsize': [256, 256],
        'img_type': 'tif'
    }
    
    # Process Train and Test videos
    for subdir_idx, subdir in enumerate(sub_dir_list):
        subdir_file_num = file_num_list[subdir_idx]
        subdir_in_path = os.path.join(in_path, subdir)
        subdir_out_path = os.path.join(out_path, subdir)
        for i in range(1, subdir_file_num + 1):
            v_name = f"{subdir}{i:03d}"
            v_path = os.path.join(subdir_in_path, v_name)
            v_out_path = os.path.join(subdir_out_path, v_name)
            utils.mkdirfunc(v_out_path)
            utils.trans_img2img(v_path, v_out_path, opts)
    
    # Generate ground truth labels for Test videos
    if 'Test' in sub_dir_list:
        gt_in_path = os.path.join(in_path, 'Test')
        gt_out_path = os.path.join(out_path, 'Test_gt')
        utils.mkdirfunc(gt_out_path)
        for i in range(1, file_num_list[1] + 1):
            v_name = f"Test{i:03d}"
            v_out_path = os.path.join(out_path, 'Test', v_name)
            # Count frames in the preprocessed output directory
            frame_list = [f for f in os.listdir(v_out_path) if f.endswith('.jpg')]
            frame_num = len(frame_list)
            gt_dir = os.path.join(gt_in_path, f"{v_name}_gt")
            if os.path.exists(gt_dir):
                # Process videos with anomalies
                utils.trans_img2label(gt_in_path, i, gt_out_path)
            else:
                # Generate all-zero labels for videos without anomalies
                l = np.zeros(frame_num, dtype=int)
                scipy.io.savemat(os.path.join(gt_out_path, f"{v_name}.mat"), {'l': l})
                print(f"Generated all-zero labels for {v_name}")
    
    print("Image preprocessing completed.")

def generate_indices():
    """Generate clip indices from preprocessed data."""
    print("Starting index generation...")
    
    # Paths
    data_root_path = '/home/westin/Documents/Code/'
    in_path = os.path.join(data_root_path, 'datasets/processed/UCSD_P2_256/')
    
    # Parameters
    frame_file_type = 'jpg'
    clip_len = 16
    skip_step = 1
    clip_rng = clip_len * skip_step - 1
    overlap_shift = clip_len - 1  # Full overlap (shift by 1 frame)
    sub_dir_list = ['Train', 'Test']
    
    # Generate indices
    for sub_dir_name in sub_dir_list:
        print(sub_dir_name)
        sub_in_path = os.path.join(in_path, sub_dir_name)
        idx_out_path = os.path.join(in_path, f"{sub_dir_name}_idx")
        utils.mkdirfunc(idx_out_path)
        v_list = [d for d in os.listdir(sub_in_path) if os.path.isdir(os.path.join(sub_in_path, d))]
        for v_name in v_list:
            print(v_name)
            frame_list = [f for f in os.listdir(os.path.join(sub_in_path, v_name)) if f.endswith(f".{frame_file_type}")]
            frame_num = len(frame_list)
            step = clip_rng + 1 - overlap_shift
            s_list = np.arange(1, frame_num + 1, step)
            e_list = s_list + clip_rng
            idx_val = e_list <= frame_num
            s_list = s_list[idx_val]
            video_sub_dir_out_path = os.path.join(idx_out_path, v_name)
            utils.mkdirfunc(video_sub_dir_out_path)
            for j, s in enumerate(s_list, 1):  # 1-based for file naming
                idx = np.arange(s, s + clip_rng + 1, skip_step)  # 1-based indices
                scipy.io.savemat(
                    os.path.join(video_sub_dir_out_path, f"{v_name}_i{j:03d}.mat"),
                    {'v_name': v_name, 'idx': idx}
                )
    
    print("Index generation completed.")

def main():
    """Main function to run preprocessing and index generation."""
    preprocess_images()
    generate_indices()

if __name__ == "__main__":
    main()