import numpy as np
import matplotlib.pyplot as plt
import sys

from Animate import generateAnimat


def main():
    width = int(sys.argv[1])
    height = int(sys.argv[2])

    learning = 0.9  # the rate at which the AI agent should learn
    epoch = 1000  # Default number of episodes
    gamma = 0.9  # Default value for gamma
    num = 3  # Default number of mines
    start = (0, 0)  # Default start state
    end = (width - 1, height - 1)  # Default end state

    # Defines all the states
    states = []
    for i in range(height):
        for j in range(width):
            states.append((j, i))

    # Reads in flags and their respective parameters from the command line
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-start":  # Sets start state
            x_start = int(sys.argv[i + 1])
            y_start = int(sys.argv[i + 2])
            start = (x_start, y_start)
        if sys.argv[i] == "-end":  # Sets end state
            x_end = int(sys.argv[i + 1])
            y_end = int(sys.argv[i + 2])
            end = (x_end, y_end)
        if sys.argv[i] == "-k":  # Sets number of mines
            num = int(sys.argv[i + 1])
        if sys.argv[i] == "-gamma":  # Sets the discount factor
            gamma = float(sys.argv[i + 1])
        if sys.argv[i] == "-learning":  # Sets the learning rate
            learning = float(sys.argv[i + 1])
        if sys.argv[i] == "-epochs":  # Specifies how many episodes your agent should learn for
            epoch = int(sys.argv[i + 1])

    # Defines the mines
    mines = []
    count = 0
    while count < num:
        x_rand = np.random.choice(width)  # Chooses a random x coordinate
        y_rand = np.random.choice(height)  # Chooses a random y coordinate
        if (x_rand, y_rand) != start and (x_rand, y_rand) != end:  # Checks it is not equal to the start or end states
            if (x_rand, y_rand) not in mines:  # Checks it is not already a terminal state
                mines.append((x_rand, y_rand))
                count += 1

    # Defines rewards for all the states
    rewards = {}
    for i in states:
        if i == end:
            rewards[i] = 100
        elif i in mines:
            rewards[i] = -100
        else:
            rewards[i] = -1

    # Possible actions for each state
    actions = ['U', 'R', 'D', 'L']

    # Defines initial numpy array containing Q values
    # Initializes all Q values as 0
    Q = np.zeros((height, width, 4))

    # Defines initial array to hold records
    records = []

    for episode in range(epoch):  # Iterates through episodes
        currS = tuple(start)  # Sets starting state
        record = [[-1 for i in range(width)] for j in range(height)]  # Record for current iteration
        while rewards[currS] == -1:  # Iterates until agent enters a terminal state
            if np.random.random() < 0.9:  # For exploration
                nextA = np.random.randint(4)
            else:  # For exploitation
                nextA = np.argmax(Q[currS[1], currS[0]])

            oldS = tuple(currS)  # Stores the old state

            # Updates the current state
            if actions[nextA] == 'U' and currS[1] > 0:
                currS = [currS[0], currS[1] - 1]
            elif actions[nextA] == 'R' and currS[0] < width - 1:
                currS = [currS[0] + 1, currS[1]]
            elif actions[nextA] == 'D' and currS[1] < height - 1:
                currS = [currS[0], currS[1] + 1]
            elif actions[nextA] == 'L' and currS[0] > 0:
                currS = [currS[0] - 1, currS[1]]

            currS = tuple(currS)
            oldQ = Q[oldS[1], oldS[0], nextA]  # Obtains Q Value for previous state

            # Calculates new Q Value
            newQ = oldQ + (learning * (rewards[currS] + (gamma * np.max(Q[currS[1], currS[0]])) - oldQ))
            Q[oldS[1], oldS[0], nextA] = newQ  # Updates old Q Value
            record[currS[1]][currS[0]] = newQ  # Updates record

        records.append(record)  # Adds record to total record

    # Define empty array for optimal policy
    opt_pol = [start]
    temp = tuple(start)
    while rewards[temp] == -1:
        # Iterates through Q, starting at the start state
        # Follows the current temp state's action until reaching the end state
        act = np.argmax(Q[temp[1], temp[0]])

        if actions[act] == 'U' and temp[1] > 0:
            temp = [temp[0], temp[1] - 1]
        elif actions[act] == 'R' and temp[0] < width - 1:
            temp = [temp[0] + 1, temp[1]]
        elif actions[act] == 'D' and temp[1] < height - 1:
            temp = [temp[0], temp[1] + 1]
        elif actions[act] == 'L' and temp[0] > 0:
            temp = [temp[0] - 1, temp[1]]

        temp = tuple(temp)
        opt_pol.append(temp)

    # Animates plot
    anim, fig, ax = generateAnimat(records, start_state=start, end_state=end, mines=mines, opt_pol=opt_pol,
                                   start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=False,
                                   vmin=-10, vmax=150)
    plt.show()


if __name__ == '__main__':
    main()
