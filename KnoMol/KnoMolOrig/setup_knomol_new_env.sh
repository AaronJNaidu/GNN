#!/bin/bash

# Create conda environment yaml file
cat > knomol_new_env.yml <<EOF
name: knomol_new
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9.10
  - pip
  - numpy=1.22.1
  - scikit-learn=1.1.2
  - scipy=1.9.1
  - rdkit=2022.03.2
  - networkx=2.8.6
  - pandas=1.4.4
  - pytorch=1.10.0
  - cudatoolkit=11.3
  - pip:
    - hyperopt==0.2.7
    - torchaudio==0.10.0
    - torchvision==0.11.0
EOF

echo "Creating conda environment knomol_new..."
conda env create -f knomol_new_env.yml

echo "Activating environment knomol_new..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate knomol_new

echo "Installing PyTorch Geometric packages..."

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric

echo "Environment setup complete. Use 'conda activate knomol_new' to activate it."
