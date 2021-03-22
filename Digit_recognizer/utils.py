
import collections
import matplotlib.pyplot as plt

def get_last_elem(x):
    '''Gets the last element of the input variable.'''

    if isinstance(x, collections.abc.Iterable):
        return x[-1]
    else:
        return x

def set_spines(ax=None):
    '''Sets the box outline around the axis ax.

    Args:
      ax: Matplotlib axes object. Defaults to the current axis.
    '''

    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
