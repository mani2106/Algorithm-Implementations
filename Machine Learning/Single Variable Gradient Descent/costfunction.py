""" Implements cost function """
import numpy as np

def calculate_cost(X: np.ndarray, y: np.ndarray, theta: float):
    """
    Calculates the error value for the
    current hypothesis
    """
    m = len(y)
    h_x = X*theta
    error = (h_x - y)
    error_sq = error * error

    return sum(error_sq)/2*m