
import collections
import matplotlib.pyplot as plt

def get_last_elem(x):
    '''Gets the last element of the input variable.'''

    if isinstance(x, collections.abc.Iterable):
        return x[-1]
    else:
        return x

def set_spines():
    '''Sets the box outline around the specified axis.'''

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
