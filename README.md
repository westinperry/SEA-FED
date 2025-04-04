# PG-FAD  
The project explores the application of federated learning techniques with a personalized approach, leveraging gating mechanisms to enhance anomaly detection across distributed systems.

<ins>[Westin Perry](mailto:wcp9372@g.rit.edu)</ins>

<div align="center">
<img src="figures/test.png" width="800px">
<p><i>Figure: Overview of the PG-FAD framework, demonstrating personalized federated anomaly detection with gated modules.</i></p>
</div>

## 📋 Table of Contents
- [PG-FAD](#pg-fad)
  - [📋 Table of Contents](#-table-of-contents)
  - [Installation](#installation)
    - [Clone the repository](#clone-the-repository)
    - [Create and activate conda environment](#create-and-activate-conda-environment)
    - [Install additional dependencies](#install-additional-dependencies)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Demo](#demo)
  - [Results & Visualization](#results--visualization)
  - [Model Card](#model-card)
  - [Citation](#citation)
  - [License](#license)

## Installation

### Clone the repository
```bash
git clone https://github.com/westinperry/PG-FAD.git
cd PG-FAD
```
### Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate PG-FAD
```
### Install additional dependencies if any
```bash
pip install -e .
```

## Project Structure

```
PG-FAD/
    ├── data/           # Scripts for managing Video Datasets
    ├── datasets/       # Dataset storage
    │   ├── processed_1   # Individual Client Data
    │   ├── processed_2
    │   └── ...
    ├── figures/        # Images for GitHub Page
    ├── models/         # Trained model checkpoints
    │   ├── client_1      # Individual Client Model
    │   ├── client_2
    │   └── ..
    ├── options/        # Script for Training/Testing Options
    ├── results/        # Testing Results
    ├── scripts/        # Scripts Training/Testing/FedAvg
    ├── utils/          # Utility Scripts
    ├── _config.yml     # GitHub Theme
    ├── commands.txt    # Useful commands
    ├── environment.yml # Conda Environment
    ├── LICENSE         # License
    └── README.md       # Project documentation
```

## Usage

### Training
To Run everything use:
```bash
cd scripts
./run.sh
```
<ul>
<li>This script will:</li>
<li></li>
<li>    Train each client (from scratch or resume from checkpoints).</li>
<li></li>
<li>    Perform federated averaging.</li>
<li></li>
<li>    Run the testing phase.</li>
<li></li>
<li>    Save all outputs (checkpoints, logs, and results) in the appropriate folders.</li>
<li>  ⚠️ Adjust the Epoch # (for each client) and Round # (Number of times clients are trained, then FedAvged (with or without gates)) </li>
</ul>

#### Direct Python Training command:
```bash
python script_training.py --DataRoot ../data --ModelRoot ../models --OutputFile final_model.pt --ModelName AE
```
(Replace --ModelName AE with --ModelName Gated_AE to use the gated model.)
AE was used as baseline.

#### Direct Python Evaluation Command:

Run the evaluation script to compute ROC curves and AUC scores:

```bash
python script_testing.py --DataRoot ../data --Dataset UCSD_P2_256 --ModelFilePath ../models/final_model.pt --ModelName AE
```
The evaluation outputs (plots, results.txt, etc.) are saved to the results/ folder (located one directory above the scripts folder).

## Demo

[Provide instructions for running the demo]

```bash
# Example demo command
python run_demo.py --input data/path/to/input
```

## Results & Visualization

[Add tables, figures, or graphs showing your key results]

| Metric | Value |
|--------|-------|
| Accuracy | X% |
| Precision | X% |
| Recall | X% |


## Citation

If you use this code in your research, please cite:

```bibtex
@article{
  westin,
  title="PG-FAD: Personalized Federated Anomaly Detection with Gated Modules",
  author="Westin Perry",
  institution="Rochester Institute of Technology",
  year="202x"
}
```
## Acknowledgments

This project extends the work presented in [Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection](https://github.com/donggong1/memae-anomaly-detection) by Dong Gong et al. Their innovative approach has provided a solid foundation and inspiration for the techniques implemented in PG-FAD. We are grateful for their contribution to the field.

Below is the citation for their work:

```bibtex
@inproceedings{gong2019memorizing,
  title={Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection},
  author={Gong, Dong and Liu, Lingqiao and Le, Vuong and Saha, Budhaditya and Mansour, Moussa Reda and Venkatesh, Svetha and Hengel, Anton van den},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
