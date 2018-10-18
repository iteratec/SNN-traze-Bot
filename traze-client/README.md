# Traze Client Python
A Traze client based on Python 3 with an example bot using a Spiking Neural Network which is simulated with [NEST](http://www.nest-simulator.org/).

## Hosted by iteratec
You can join a hosted game instance at [traze.iteratec.de](https://traze.iteratec.de).

## Installation on Unix (tested with Ubuntu 18.04.1 LTS)

We recommend installing using the install file:
```
source install.sh

#start the bot
python ./bots/SNNBot.py <minutes_until_reset>
```

If you prefer a manual installation, follow the steps below:

Install [Miniconda](https://conda.io/miniconda.html) with Python 3.7 and execute the following commands:
```
# Create and activate a virtual environment
conda create --name traze python=3.7
source activate traze

# Install cython
conda install cython
```

Install NEST version 2.16.0:
1. Clone the repository: `git clone --branch v2.16.0 --depth 1 https://github.com/nest/nest-simulator.git`
2. Create build directory: `mkdir nest-simulator/build`
3. Change to build directory: `cd nest-simulator/build`
4. Configure NEST: `cmake -Dwith-python=3 -DCMAKE_INSTALL_PREFIX:PATH=</build/path> ..` (/build/path should be the absolute path to nest-simulator/build)
5. Compile and install by running `make install`
6. Set environment variables for NEST with `source </build/path>/bin/nest_vars.sh`
or source it to .profile with `cat </build/path>/bin/nest_vars.sh >> ~/.profile`.

Install the requirements and run the bot:
```
cd <path/to/SNN-Traze>
pip install -r ./traze-client/requirements.txt
python ./traze-client/bots/SNNBot.py <minutes_until_reset>
```

The bot automatically connects to [traze.iteratec.de](https://traze.iteratec.de/watch) and starts playing. The <minutes_until_reset> argument defines the number of minutes until the bot resets its weights.

## Structure of the Spiking Neural Network
Three input neurons are fully connected to three output neurons. The input is the distance of the first objects in the directions left, forward and right from the current position.
The output neurons each represent an action (left, forward, right) and the bot chooses the action from the output neuron, which received the most spikes.