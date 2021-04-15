import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from time import time


# Misc
def get_last_elem(x):
    """Gets the last element of the input variable."""

    if np.isscalar(x):
        return x
    else:
        return x[-1]


# Plot
wong_colors = {
    'blue': [0,114/255,178/255],
    'orange': [230/255,159/255,0],
    'bluish_green': [0,158/255,115/255],
    'vermillion': [213/255,94/255,0],
    'sky_blue': [86/255,180/255,233/255],
    'yellow': [240/255,228/255,66/255],
    'black': [0,0,0]
}

def set_spines_vis(types=['top', 'right'], visible=False, ax=None):
    """
    Sets the spines' visibility.

    Parameters
    ----------
    types : {'left', 'right', 'bottom', 'top'}
        Spines types.
    visible : bool
        Spines' visibility.
    ax : matplotlib axes object
       The axes object containing the spine.
    """

    # Set axes object default value
    if ax is None:
        ax = plt.gca()

    # Set the spines' visibility
    for val in types:
        ax.spines[val].set_visible(visible)


# Pandas
def df_info(df):
    """
    Returns a DataFrame with the data type and null count of each column.

    Parameters
    ----------
    df : DataFrame
        A Pandas DataFrame.

    Returns
    -------
    info : DataFrame
        DataFrame with the information of df.
    """

    info = pd.DataFrame({'Null count': df.isnull().sum(), 'Dtype': df.dtypes})

    return info

# Machine learning
def create_models(create_fun, create_params):
    """
    Creates ML models based on create_params.

    Parameters
    ----------
    create_fun : function
        Function to use to create the models.
    create_params : dict
        Dictionary with parameters names (str) as keys and respective values. Must have exactly one
        list or 1-D NumPy array that corresponds to the parameter that varies between models. All
        other parameters are kept constant.

    Returns
    -------
    models : list
        Created models.
    param_name : str
        Name of the parameter that varies between models.
    param_vals : list or 1-D NumPy array
        Values of the parameter that varies between models.
    """

    # Get the parameter that has a list or NumPy array of values
    keys = list(create_params.keys())
    is_valid = [isinstance(create_params[k], (list, np.ndarray)) for k in keys]

    if sum(is_valid) != 1:
        raise ValueError("create_params must have exactly one list or NumPy array")

    param_name = keys[is_valid.index(True)]
    param_vals = create_params[param_name]

    # Create the models
    create_params = create_params.copy()
    del create_params[param_name]
    models = []

    for val in param_vals:
        models.append(create_fun(**create_params, **{param_name:val}))

    return models, param_name, param_vals

def sklearn_fit_eval(model, fit_params, X_val=None, y_val=None, cv=None, verbose=True):
    """
    Fits a Scikit-learn model to the train set and evaluates its accuracies and fitting time.

    Parameters
    ----------
    model : estimator object implementing 'fit'
        Model to fit.
    fit_params : dict
        Parameters to pass to the fit method of the model.
    X_val : array-like
        Input data used to calculate the validation accuracy.
    y_val : array-like
        Target vector used to calculate the validation accuracy.
    cv : int
        Number of folds in a k-folds cross-validation.
    verbose : bool
        Controls verbosity of output.

    Returns
    -------
    train_accuracy : float
        Train accuracy.
    val_accuracy : float
        Validation accuracy.
    time : float
        Time for fitting the model on the train set.
    """

    # Fit the model
    start = time()
    model.fit(**fit_params)
    fit_time = time() - start

    # Calculate the train accuracy
    X_train = fit_params['X']
    y_train = fit_params['y']
    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    # Calculate the validation accuracy
    if X_val is not None:
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
    elif cv is not None:
        val_accuracy = cross_validate(model, X_train, y_train, cv=cv)['test_score'].mean()
    else:
        val_accuracy = None

    # Print information about the fitting process
    if verbose:
        print('Train accuracy: {:.2%}'.format(train_accuracy), end='')
        if val_accuracy != None:
            print('; Validation accuracy: {:.2%}'.format(val_accuracy), end='')
        print('; Fitting time: {:.2f}s'.format(fit_time))

    return train_accuracy, val_accuracy, fit_time

def plot_models_metrics(create_fun, create_params, fit_eval_fun, fit_eval_params, fig_params={},
                        axes_params={}):
    """
    Creates and fits the models and plots their metrics.

    Parameters
    ----------
    create_fun : function
        Function to use to create the models.
    create_params : dict
        Parameters to pass to create_fun. Must have exactly one array-like value that corresponds
        to the parameter that varies between models. All other parameters are the same in all
        models.
    fit_eval_fun : function
        Function to use to fit and evaluate the models.
    fit_eval_params : dict
        Parameters to pass to fit_eval_fun.
    fig_params : dict
        Parameters to pass to the function that creates the figure.
    axes_params : dict
        Parameters to pass to the function that creates the axes.
    """

    # Create the models
    models, param_name, param_vals = create_models(create_fun, create_params)

    # Set default values of fig_params
    if 'figsize' not in fig_params:
        fig_params['figsize'] = (15, 4)

    # Fit the models and calculate their metrics
    train_accuracies = []
    val_accuracies = []
    fit_times = []

    for model in models:
        outputs = fit_eval_fun(model, **fit_eval_params)
        train_accuracies.append(get_last_elem(outputs[0]))
        val_accuracies.append(get_last_elem(outputs[1]))
        fit_times.append(outputs[2])

    # Check if the parameters values are numbers
    if isinstance(param_vals[0], numbers.Number):
        x = param_vals
    else:
        x = range(len(param_vals))

    # Plot the metrics
    plt.figure(**fig_params)

    plt.subplot(1, 2, 1, **axes_params)
    plt.plot(x, train_accuracies)
    plt.plot(x, val_accuracies)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], frameon=False)
    set_spines_vis()

    plt.subplot(1, 2, 2, **axes_params)
    plt.plot(x, fit_times)
    plt.xlabel(param_name)
    plt.ylabel('Fitting time (s)')
    set_spines_vis()

def sklearn_plot_losses(create_fun, create_params, fit_params={}, legend=None, fig_params={}):
    """
    Creates and fits the models and plots their loss curves.

    Parameters
    ----------
    create_fun : function
        Function to use to create the models.
    create_params : dict
        Parameters to pass to create_fun. Must have exactly one array-like value that corresponds
        to the parameter that varies between models. All other parameters are the same in all
        models.
    fit_params : dict
        Parameters to pass to the fit method of the models.
    legend : list
        Legend to place on the axes.
    fig_params : dict
        Parameters to use to create the figure.
    """

    # Create the models
    models, _, param_vals = create_models(create_fun, create_params)

    # Set default legend
    if legend is None:
        legend = [str(val) for val in param_vals]

    # Set default values of fig_params
    if 'figsize' not in fig_params:
        fig_params['figsize'] = (10, 4)

    # Fit the models and plot their loss curves
    plt.figure(**fig_params)

    for i, model in enumerate(models):
        start = time()
        model.fit(**fit_params)
        fit_time = time() - start

        plt.plot(model.loss_curve_)

        args = (legend[i], model.loss_curve_[-1], fit_time)
        print('Value: {}; Loss: {:.2f}; Fitting time: {:.2f}s'.format(*args))

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(legend, frameon=False)
    set_spines_vis()
