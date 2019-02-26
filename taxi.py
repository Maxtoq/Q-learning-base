import numpy as np
import random as rd

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class Taxi:

    R_STEP = -1
    R_GOOD_PICKUP = 10
    R_BAD_PICKUP = -10
    R_GOOD_DROPOFF = 20
    R_BAD_DROPOFF = -10

    def __init__(self):
        self.map = np.asarray(MAP)
        self.colsize = 5
        self.rowsize = 5

        # The four locations (starting and ending points)
        self.locs = [(0,0), (0,4), (4,0), (4,3)]

        # Taxi, with random coordinates
        self.taxi = [0, 0]
        while (self.taxi[0], self.taxi[1]) in self.locs:
            self.taxi = [rd.randint(0, 4), rd.randint(0, 4)]

        # 5 different locations: 0, 1, 2, 3 for the four starting points, 4 for in the taxi
        self.humanloc = rd.randint(0, 3)
        # 4 different finishing points, different from starting point
        self.humangoal = self.humanloc
        while self.humangoal == self.humanloc:
            self.humangoal = rd.randint(0, 3)

        # State
        self.currstate = self.getState(self.taxi, self.humanloc)

        # Action
        self.action = -1

        # Finished episode
        self.done = False

        # Hyperparameters
        self.e = 1.0 # Exploration rate, epsilon, 1 at the beginning to explore random actions
        self.min_e = 0.01
        self.decay_rate = 0.001

        self.lr = 0.1 # Learning rate
        self.dr = 0.9 # Discount rate

        self.nb_train_episode = 5000 # Number of training episodes
        self.nb_max_steps = 100 # Maximum number of steps during an episode

        # Q table
        # 500 discrete states:
        #   25 taxi positions,
        #   5 possible locations of the human (including the case when the passenger is the taxi),
        #   4 destination locations
        # 6 possible actions:
        #   0: move south
        #   1: move north
        #   2: move east
        #   3: move west
        #   4: pickup passenger
        #   5: dropoff passenger
        self.qt = np.zeros((500, 6))

    def display(self):
        """ Displays the map, putting the human and the taxi at their location. """
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if row == self.taxi[0] + 1 and col == 2 * self.taxi[1] + 1:
                    if self.humanloc != 4:
                        print('T', end='')
                    else:
                        print('F', end='')
                elif self.humanloc != 4 and row == self.locs[self.humanloc][0] + 1 and col == 2 * self.locs[self.humanloc][1] + 1:
                    print('H', end='')
                else:
                    print(self.map[row][col], end='')
            print('')

    def init(self):
        # Taxi, with random coordinates
        self.taxi = [0, 0]
        while (self.taxi[0], self.taxi[1]) in self.locs:
            self.taxi = [rd.randint(0, 4), rd.randint(0, 4)]

        # 5 different locations: 0, 1, 2, 3 for the four starting points, 4 for in the taxi
        self.humanloc = rd.randint(0, 3)
        # 4 different finishing points, different from starting point
        self.humangoal = self.humanloc
        while self.humangoal == self.humanloc:
            self.humangoal = rd.randint(0, 3)

        # State
        self.currstate = self.getState(self.taxi, self.humanloc)

        # Action
        self.action = -1

        # Finished episode
        self.done = False

    def getState(self, pos, loc):
        """ Returns the number of the given state. """
        return (5 * pos[0] + pos[1]) * 20 + loc * 4 + self.humangoal

    def exploration(self):
        """ Do the Exploration policy. """
        self.action = rd.randint(0, 5)

    def exploitation(self):
        """ Do the Exploitation policy. """
        # Look for the best action possible in the current state
        qmax = np.amax(self.qt[self.currstate])

        # If this max is the q-value for multiple actions, choose randomly between these actions
        count = np.count_nonzero(self.qt[self.currstate] == qmax)

        if count > 1:
            rn = rd.randint(1, count + 1)
            act = 0
            for i in range(0, len(self.qt[self.currstate])):
                if self.qt[self.currstate, i] == qmax:
                    act = i
                    rn -= 1
                if rn == 0:
                    break
            self.action = act
        else:
            self.action = np.argmax(self.qt[self.currstate])

    def chooseAction(self):
        """ Choose an action to do. """
        # Decide wether we'll do Exploration or Exploitation
        n = rd.random()

        if n > self.e:
            # Do Exploitation
            self.exploitation()
        else:
            # Do Exploration
            self.exploration()

    def takeAction(self):
        """ Take the chosen action, and evaluate it. """
        # New state
        newtaxi = [0, 0]
        newtaxi[0] = self.taxi[0]
        newtaxi[1] = self.taxi[1]

        newloc = self.humanloc

        # Reward
        r = self.R_STEP

        # Go South
        if self.action == 0:
            if self.taxi[0] < self.rowsize - 1:
                newtaxi[0] += 1
        # Go North
        elif self.action == 1:
            if self.taxi[0] > 0:
                newtaxi[0] -= 1
        # Go East
        elif self.action == 2:
            if self.taxi[1] < self.colsize - 1 and self.map[self.taxi[0] + 1][2 * self.taxi[1] + 2] == ':':
                newtaxi[1] += 1
        # Go West
        elif self.action == 3:
            if self.taxi[1] > 0 and self.map[self.taxi[0] + 1][2 * self.taxi[1]] == ':':
                newtaxi[1] -= 1
        # Pickup
        elif self.action == 4:
            if self.humanloc < 4 and (self.taxi[0], self.taxi[1]) == self.locs[self.humanloc]:
                newloc = 4
                r += self.R_GOOD_PICKUP
            else:
                r += self.R_BAD_PICKUP
        # Dropoff
        elif self.action == 5:
            if self.humanloc == 4 and (self.taxi[0], self.taxi[1]) == self.locs[self.humangoal]:
                newloc = self.humangoal
                r += self.R_GOOD_DROPOFF
                self.done = True
            elif self.humanloc == 4 and (self.taxi[0], self.taxi[1]) in self.locs[self.humangoal]:
                newloc = self.locs.index((self.taxi[0], self.taxi[1]))
                r += self.R_BAD_DROPOFF
            else:
                r += self.R_BAD_DROPOFF

        # Get new state
        newstate = self.getState(newtaxi, newloc)

        # Evaluate
        self.qt[self.currstate, self.action] += self.lr * (r + self.dr * np.amax(self.qt[newstate]) - self.qt[self.currstate, self.action])

        # Change values
        self.taxi[0] = newtaxi[0]
        self.taxi[1] = newtaxi[1]
        self.currstate = newstate
        self.humanloc = newloc

    def episode(self, show=False):
        """ Do one episode. """
        if show:
            self.display()

        for i in range(0, self.nb_max_steps):
            # Choose action
            self.chooseAction()

            # Take action and evaluate it
            self.takeAction()

            if show:
                self.display()

            if self.done == True:
                print("WIN", i)
                break

        if self.done == False:
            print("LOSE")

        # Decay Epsilon
        if self.e > self.min_e:
            self.e -= self.decay_rate

    def train(self):
        """ Trains the taxi. """
        for i in range(0, self.nb_train_episode):
            print("Episode:", i + 1, end=' ')
            self.episode()

            self.init()

    def result(self):
        """ Show results of training. """
        self.e = 0

        self.init()

        self.episode(show=True)


# Create Ranjit, the taxi driver
ranjit = Taxi()
ranjit.display()

# Train Ranjit
ranjit.train()

ranjit.result()
