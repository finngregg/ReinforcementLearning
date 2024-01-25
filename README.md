## ValueIteration.py
ValueIteration.py implements an algorithm which is capable of determining the best policy (path) for an agent traversing a grid. 
This file contains its main method, which implements the necessary algorithms to determine the agents optimal policy. 
The parameters of this grid can be entered via the command line. 
It imports Animate.py in order to animate the multiple iterations that are run and recorded when determining this policy. 

-start xpos ypos (start co-ordinate)

-end xpos ypos (end co-ordinate)

-k num (number of mines)

-gamma g (discount factor)

* Methods
```python
    def main():
```

## QLearning.py
QLearning.py implements the Reinforcement Learning Algorithm, allowing it to calculate the Q value of a particular state based on the states surrounding environment.
This file contains its main method, which implements the necessary methods to not only determine the Q values of each state, but find the shortest path of a grid from start to finish.
The parameters of this grid can be entered via the command line. 
It imports Animate.py in order to animate the multiple iterations and final solution.

-start xpos ypos (start co-ordinate)

-end xpos ypos (end co-ordinate)

-k num (number of mines)

-gamma g (discount factor)

-learning n (learning rate)

-epochs e (number of episodes)

* Methods
```python
    def main():
```

## Makefile
The Makefile builds a working python virtual environment from scratch. 

Some examples have been commented out in the Makefile. The main commands include:
* make - creates virtual environment
* make clean - removes virtual environment

The environment must first be built using 'make' and then activated with 'source ./venv/bin/activate'
