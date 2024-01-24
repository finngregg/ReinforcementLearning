import numpy as np
import matplotlib.pyplot as plt
import sys

from Animate import generateAnimat


def main():
    width = int(sys.argv[1])
    height = int(sys.argv[2])

    theta = 0.005  # Minimum value that allows for convergence
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

    # Defines the mines
    mines = []
    count = 0
    while count < num:
        x_rand = np.random.choice(width)  # Chooses a random x coordinate
        y_rand = np.random.choice(height)  # Chooses a random y coordinate
        if (x_rand, y_rand) != start and (
        x_rand, y_rand) != end:  # Checks the state is not equal to the start or end states
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
            rewards[i] = 0

    # Dictionary of possible actions based on a state's location
    actions = {}
    for i in states:
        if i in mines or i == end:
            continue
        else:
            if i[1] == 0:  # Possible conditions and actions if y = 0 i.e. the top
                if i[0] == 0:
                    actions[i] = ('D', 'R')
                elif i[0] == width - 1:
                    actions[i] = ('D', 'L')
                else:
                    actions[i] = ('D', 'L', 'R')
            elif i[1] == height - 1:  # Possible conditions and actions if y = height-1 i.e the bottom
                if i[0] == 0:
                    actions[i] = ('U', 'R')
                elif i[0] == width - 1:
                    actions[i] = ('U', 'L')
                else:
                    actions[i] = ('U', 'L', 'R')
            elif i[0] == 0:  # Possible actions if x = 0
                actions[i] = ('U', 'D', 'R')
            elif i[0] == width - 1:  # Possible actions if x = width-1
                actions[i] = ('U', 'D', 'L')
            else:
                actions[i] = ('U', 'D', 'L', 'R')  # Possible actions if state is not on a boundary

    # Defines an initial policy
    policy = {}
    for s in actions.keys():
        policy[s] = np.random.choice(actions[s])

    # Defines the initial value function
    V = {}
    for s in states:
        if s in actions.keys():
            V[s] = 0
        if s in mines:
            V[s] = -100
        if s == end:
            V[s] = 100

    # Defines initial array to hold records
    records = []

    iteration = 0
    while True:
        delta = 0
        curr = [[0 for i in range(width)] for j in range(height)]  # Record for current iteration
        for state in states:
            if state in policy:

                oldV = V[state]
                newV = 0

                # Updates the state
                for action in actions[state]:
                    if action == 'U':
                        nextS = [state[0], state[1] - 1]
                    if action == 'D':
                        nextS = [state[0], state[1] + 1]
                    if action == 'L':
                        nextS = [state[0] - 1, state[1]]
                    if action == 'R':
                        nextS = [state[0] + 1, state[1]]

                    nextS = tuple(nextS)

                    # Calculate the value of the state
                    v = rewards[state] + (gamma * V[nextS])
                    if v > newV:
                        newV = v  # Sets the state's value to the updated value
                        policy[state] = action  # Sets the appropriate/best action for the state

                # Save the best of all actions for the state
                curr[state[1]][state[0]] = newV  # Adds value to current iterations record
                V[state] = newV
                delta = max(delta, np.abs(oldV - V[state]))

            else:
                curr[state[1]][state[0]] = V[state]

        records.append(curr)  # Adds current iteration's record to total record

        # Check whether convergence has been reached
        if delta < theta:
            break

        # Increment iteration
        iteration += 1

    # Define empty array for optimal policy
    opt_pol = [start]
    temp = tuple(start)
    while True:
        # Iterates through the policy, starting at the start state
        # Follows the current state's action until reaching the end state
        if temp == end:
            break
        elif policy[temp] == 'U':
            temp = tuple([temp[0], temp[1] - 1])
        elif policy[temp] == 'D':
            temp = tuple([temp[0], temp[1] + 1])
        elif policy[temp] == 'L':
            temp = tuple([temp[0] - 1, temp[1]])
        elif policy[temp] == 'R':
            temp = tuple([temp[0] + 1, temp[1]])

        opt_pol.append(temp)

    # Animates plot
    anim, fig, ax = generateAnimat(records, start_state=start, end_state=end, mines=mines, opt_pol=opt_pol,
                                   start_val=-10, end_val=100, mine_val=150, just_vals=False, generate_gif=False,
                                   vmin=-10, vmax=150)
    plt.show()


if __name__ == '__main__':
    main()
