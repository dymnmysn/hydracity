#!/usr/bin/env bash
# Updated Script to Include PyTorch 2.4 Support for Kaggle

NNDCT_REPO="${NNDCT_REPO:-https://github.com/Xilinx/Vitis-AI.git}"
NNDCT_SRC="${NNDCT_SRC:-src/vai_quantizer/vai_q_pytorch}"
BRANCH="${BRANCH:-master}"

valid_torch_version=("1.4" "1.7.1" "1.8.0" "1.12.1" "2.0.0" "2.4.0") # Add PyTorch 2.4 here
if [ -z "$1" ]; then
  echo "Usage: $0 torch_version"
  echo "  Current valid torch versions supported by this script are: ${valid_torch_version[*]}"
  exit 2
fi
IFS=',' read -r -a torch_version <<< "$1"

for torch_input in ${torch_version[@]}; do
   if [[ ! " ${valid_torch_version[*]} " =~ " ${torch_input} " ]]; then
      echo "${torch_input} is not supported. Update the script for this version."
      exit 2
   fi
done

# Create Conda environment and configure for PyTorch
for torch_to_install in ${torch_version[@]}; do
  echo -e "\n#### Configuring for PyTorch $torch_to_install..."

  # Skip PyTorch installation if using Kaggle's pre-installed version
  if [[ $torch_to_install == "2.4.0" ]]; then
    echo "Using Kaggle's pre-installed PyTorch 2.4."
  else
    install_version=""
    if [ -d "/usr/local/cuda" ]; then
      # GPU Installations
      case $torch_to_install in
        "2.4.0")
          install_version="torch==2.4.0 torchvision==0.16.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118"
          ;;
        # Other cases remain unchanged
      esac
    else
      # CPU Installations
      case $torch_to_install in
        "2.4.0")
          install_version="torch==2.4.0+cpu torchvision==0.16.0+cpu torchaudio==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu"
          ;;
        # Other cases remain unchanged
      esac
    fi

    # Install PyTorch
    if [ -n "$install_version" ]; then
      pip install $install_version
      if [ $? -ne 0 ]; then
        echo "Failed to install PyTorch $torch_to_install. Exiting."
        exit 2
      fi
    fi
  fi

  # Continue with vai_q_pytorch setup
  echo -e "\n#### Installing vai_q_pytorch..."
  cd /scratch/code_vaiq/$NNDCT_SRC/pytorch_binding
  if [ ! -d "/usr/local/cuda" ]; then
    unset CUDA_HOME
  fi
  python setup.py bdist_wheel -d ./
  pip install ./pytorch_nndct-*.whl
done
