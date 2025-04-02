# PG-FAD
The project explores the application of federated learning techniques with a personalized approach, leveraging gates to enhance anomaly detection across distributed systems.

<ins>[Westin Perry](mailto:wcp9372@g.rit.edu)</ins>

<div align="center">
<img src="figures/test.png" width="800px">
<p><i>Figure:Generated samples from Visual AutoRegressive (VAR) transformers trained on ImageNet. We show 512×512 samples (top), 256×256 samples (middle), and zero-shot image editing results (bottom).</i></p>
</div>
  
## 📋 Table of Contents
- [PG-FAD](#pg-fad)
  - [📋 Table of Contents](#-table-of-contents)
  - [Installation](#installation)
    - [Clone the repository](#clone-the-repository)
    - [Create and activate conda environment](#create-and-activate-conda-environment)
    - [Install additional dependencies if any](#install-additional-dependencies-if-any)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Demo](#demo)
  - [Results \& Visualization](#results--visualization)
  - [Model Card](#model-card)
  - [Citation](#citation)
  - [License](#license)

## Installation
### Clone the repository
```bash
git clone https://github.com/github_repo.git
cd pg_fad
```
### Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate pg_fad
```
### Install additional dependencies if any
```bash
pip install -e .
```

## Project Structure

```
pg_fad/
    ├── data/          # Dataset storage and data files
    ├── scripts/       # Standalone scripts and utilities
    ├── models/        # Trained model checkpoints
    ├── notebooks/     # Jupyter notebooks for analysis
    ├── results/       # Experimental results and metrics
    ├── figures/       # Project figures and visualizations
    └── README.md      # Project documentation
```

## Usage

### Training

```bash
python train.py
# bash train.sh
```

### Evaluation

```bash
python eval.py  --checkpoint models/path/to/checkpoint
# bash eval.sh
```

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

## Model Card
Please see the [Hugging Face page](https://huggingface.co/username/my-model)
for the full model card.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{
  westin,
  title="PG-FAD",
  author="Westin Perry",
  institution="Rochester Institute of Technology",
  year="202x"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
