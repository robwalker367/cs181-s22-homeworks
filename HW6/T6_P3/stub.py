# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    def __init__(self, _gamma = 0.9, _init_epsilon = 0.1, _init_alpha = 0.9):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.A = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))
        self.gamma = _gamma
        self.init_epsilon = _init_epsilon
        self.init_alpha = _init_alpha


        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, state):
        new_state = state
        sp = self.discretize_state(new_state)

        if self.last_state:
            s = self.discretize_state(self.last_state)
            a = self.last_action
            self.A[a, s[0], s[1]] += 1
            alpha = 1 / (1 + 0.05 * self.A[a, s[0], s[1]]) * self.init_alpha
            self.Q[a, s[0], s[1]] = self.Q[a, s[0], s[1]] + alpha * (self.last_reward + self.gamma * (np.max(self.Q[:, sp[0], sp[1]])) - self.Q[a, s[0], s[1]])
        
        new_action = 0
        epsilon = 1 / (1 + 0.5 * self.A[:, sp[0], sp[1]].sum()) * self.init_epsilon
        if npr.rand() < epsilon:
            # Take random choice of 0 or 1
            new_action = int(0.5 > npr.rand())
        else:
            # Take optimal action
            new_action = int(np.argmax(self.Q[:, sp[0], sp[1]]))

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 0)
    print(hist)

