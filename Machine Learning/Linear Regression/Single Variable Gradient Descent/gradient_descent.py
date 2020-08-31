""" Implements Gradient Descent """
import numpy as np
import costfunction as cf

def gradient_descent(X: np.ndarray,
                     y: np.ndarray,
                     theta: float,
                     alpha: float,
                     iters: int):
    '''
    Implements gradient descent to the update given theta

    Parameters
    ----------
    X : Numpy ndarray
        Training data
    y : Numpy ndarray
        Target data
    theta : float
        The Parameter to update
    alpha : float
        Learning rate
    iters : int
        Number of iterations to run
    '''
    m = len(y)
    history = []
    for _ in range(iters):
        h_x = np.dot(theta, X)
        errors = h_x - y

        # Update the parameter
        theta_ch = (alpha * np.dot(X.conj().T, errors))/m

        theta = theta - theta_ch

        # Capture history for plotting
        history.append((cf.calculate_cost(X, y, theta), theta))
    
    return history, theta
