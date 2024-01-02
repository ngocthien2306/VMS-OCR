#!/bin/bash

# Get the current username
CURRENT_USER=$(whoami)

# Function to check if a Git repository exists
repository_exists() {
  [ -d "$1" ] && [ -n "$(ls -A $1)" ]
}

# Function to clone a Git repository if it doesn't exist
clone_if_not_exists() {
  REPO_DIR="$1"
  REPO_URL="$2"

  if repository_exists "$REPO_DIR"; then
    echo "Repository in $REPO_DIR already exists. Skipping clone."
  else
    mkdir -p "$REPO_DIR"
    cd "$REPO_DIR"
    git clone "$REPO_URL"
  fi
}

# Generate SSH key
ssh-keygen -t rsa -b 4096 -C "ngocthien.dev23@gmail.com"
cat ~/.ssh/id_rsa.pub

# Clone other repositories if needed
clone_if_not_exists "/home/$CURRENT_USER/Downloads/repos/vision" "https://github.com/pytorch/vision.git"
clone_if_not_exists "/home/$CURRENT_USER/Downloads/repos/resizeSwapMemory" "https://github.com/JetsonHacksNano/resizeSwapMemory.git"

# Install Jetpack
sudo apt update
sudo apt install -y nvidia-jetpack
sudo apt update && sudo apt upgrade

cd resizeSwapMemory
./setSwapMemorySize.sh -g 4

cd ..

# Basic install
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y git cmake python3-dev nano
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
sudo apt-get install -y python3-pip
sudo pip3 install -U pip testresources setuptools

# Create virtual environment
sudo pip install virtualenv virtualenvwrapper
export WORKON_HOME="/home/$CURRENT_USER/.virtualenvs"
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh

source ~/.bashrc
mkvirtualenv ppe -p python3
workon ppe

sudo chown -R "$CURRENT_USER":"$CURRENT_USER" /home/"$CURRENT_USER"/.virtualenvs/ppe

# Install jtop
sudo apt-get install -y python3-pip
sudo -H pip install -U jetson-stats

# Build Qt5
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip python3-pyqt5.qtsvg python3-pyqt5.qtwebkit
sudo apt-get install -y qt5-default

# Install PyQt5
wget https://files.pythonhosted.org/packages/8c/90/82c62bbbadcca98e8c6fa84f1a638de1ed1c89e85368241e9cc43fcbc320/PyQt5-5.15.0.tar.gz

wget https://www.riverbankcomputing.com/static/Downloads/sip/4.19.25/sip-4.19.25.tar.gz
tar -xvzf sip-4.19.25.tar.gz
cd sip-4.19.25
python3 configure.py
make
sudo make install

cd ..
tar -xvzf PyQt5-5.15.0.tar.gz
cd PyQt5-5.15.0
python3 configure.py --qmake /usr/bin/qmake
make
sudo make install
cd ..

pip install PyQt5-sip -U

# Install PyTorch
sudo apt-get -y update
sudo apt-get install -y python3-pip libopenblas-dev
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

python3 -m pip install --upgrade pip
python3 -m pip install numpy
python3 -m pip install --no-cache $TORCH_INSTALL

cd "/home/$CURRENT_USER/Downloads/repos/vision"
git checkout v0.15.1
USE_CUDA=1 pip install -v -e . --no-use-pep517

cd ..
# Install Jetson.GPIO
pip install Jetson.GPIO
sudo groupadd -f -r gpio
sudo usermod -a -G gpio "$CURRENT_USER"
sudo cp "/home/$CURRENT_USER/.virtualenvs/ppe/lib/python3/site-packages/Jetson/GPIO/99-gpio.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Install additional packages
pip install ultralytics opencv-python pydantic-1.8.2 python-dotenv terminaltables

# Disable sleep
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo apt purge xscreensaver


# Clone PPE app repository at the end
PPE_REPO_DIR="/home/$CURRENT_USER/Downloads/repos/ppe-app"
clone_if_not_exists "$PPE_REPO_DIR" "git@github.com:ngocthien2306/ppe-app.git"
cd "$PPE_REPO_DIR"
git checkout v0.3-auto-vertical

# Run the PPE app
python main.py

# Install VSCode
sudo snap install codium --classic
