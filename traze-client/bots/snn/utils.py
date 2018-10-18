import numpy as np
import matplotlib.pyplot as plt

class AgentStatistics(object):    
    """ Represents the summary of the agent's performance. """

    def __init__(self):
        self.reset()

    def reset(self):
        self.w = []
    
    def append(self, w=None):
        self.w.append(np.array(w).reshape(-1))

    def plot(self):
        plt.plot(self.w)
        plt.show()


def print_me(array, style='.2f'):
    return str([format(x,style) for x in np.array(array).reshape(-1)]).replace('\'','')
