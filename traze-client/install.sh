#!/bin/bash

PYTHON_VERSION=3.7
NEST_VERSION=2.16.0
ENV_NAME=traze

sudo apt-get update && sudo apt-get install -y build-essential cmake wget

echo "Should a new version of Miniconda be installed to your system?"
read -p "Press the key 'n' only if you already have it! [[Y]/n] " -n 1 -r && echo # move to a new line
if [[ ! $REPLY = [Nn] ]]; then
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
	rm -f Miniconda3-latest-Linux-x86_64.sh
fi

conda create -yn ${ENV_NAME} python=${PYTHON_VERSION}
source activate ${ENV_NAME}

conda install -y cython

wget -qO- https://github.com/nest/nest-simulator/archive/v${NEST_VERSION}.tar.gz | tar xvz
mv nest-simulator-${NEST_VERSION} nest-simulator && cd nest-simulator

mkdir build && cd build
cmake -Dwith-python=3 -DCMAKE_INSTALL_PREFIX:PATH=$PWD ..
make install
cat $PWD/bin/nest_vars.sh >> ~/.bashrc && source ~/.bashrc

cd ../../
source activate ${ENV_NAME}
pip install -r requirements.txt

pip install -e git+https://github.com/iteratec/traze-client-python.git#egg=traze
