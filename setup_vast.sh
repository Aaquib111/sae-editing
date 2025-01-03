#!/bin/bash

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
# Add conda to path for this session
export PATH="$HOME/miniconda3/bin:$PATH"

# Initialize conda in the shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# Create environment
conda create -y -n sae python=3.12

# Activate environment
conda activate sae

# Verify we're in the right environment
which python
python --version
echo "Current conda environment: $CONDA_DEFAULT_ENV"

git config --global user.name "magikarp01"
git config --global user.email "philliphguo@gmail.com"

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --upgrade sae-lens
python -m ipykernel install --user --name=sae
conda install -y -c conda-forge ipywidgets jupyterlab_widgets nodejs

git clone https://github.com/magikarp01/tasks.git

# Source the .env file
set -a
source .env


# Login to HuggingFace more securely
huggingface-cli login --token $HF_ACCESS_TOKEN

# Login to WandB more securely (using their recommended environment variable method)
# WandB automatically picks up WANDB_API_KEY from environment, no need to pass it explicitly
wandb login

# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# echo "Setting up VSCode..."
# # Defining the vscode variable with the path to the VSCode executable
# vscode_path=$(ls -td ~/.vscode-server/bin/*/bin/remote-cli/code | head -1)
# vscode="$vscode_path"

# # Append vscode path to .bashrc for future use
# echo 'alias code="'$vscode'"' >> ~/.bashrc
# echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc

# # Update the system and install jq
# sudo apt-get update
# sudo apt-get install -y jq

# # Install recommended VSCode extensions
# jq -r '.recommendations[]' ~/.vscode-server/extensions/extensions.json | while read extension; do "$vscode" --install-extension "$extension"; done