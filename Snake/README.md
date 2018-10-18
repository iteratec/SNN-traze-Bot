# Snake
Use the Makefile to install the requirements and start playing, training or testing in 1D or 2D. The following commands can be used:

* `make install` to install all the requirements for Snake
* `make train` to train the SNN for the in the Makefile specified number of games in 1D
* `make play` to let the bot play for the in the Makefile specified number of games in 1D
* `make play-human` to play Snake in 1D yourself using the arrow keys.
* `make train2D` to train the SNN for the in the Makefile specified number of games in 2D
* `make play2D` to let the bot play for the in the Makefile specified number of games in 2D
* `make play-human2D` to play Snake in 2D yourself using the arrow keys.
* `make test2D` to train the SNN for the in the Makefile specified number of episodes and number of runs in 2D
* `make clean` to delete all pycache and weights (.h5) files.

## Structure of the Spiking Neural Network for 2D
Six input neurons are fully connected to three output neurons. The input is the distance of the first objects in the directions left, forward and right from the current position and the distance from the fruit in all directions.
The output neurons each represent an action (left, forward, right) and the bot chooses the action from the output neuron, which received the most spikes.