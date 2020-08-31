import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter 

blue = 'tab:blue'
red = 'tab:red'

def plot_data(hist, y, theta):
    """
    Plots static image of the error value and theta and the last fitted line
    """
    # Initialize data
    error_hist = [d[0] for d in hist]
    theta_hist = [d[1] for d in hist]
    x_dat = range(len(error_hist))
    x_d = range(len(y))

    # Initialize plotting layout
    fig, axs = plt.subplots(ncols=2, nrows=1)

    ax1 = axs[0]

    # For Errors
    color = red
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Error value', color=color)
    ax1.plot(x_dat, error_hist, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # For theta
    color = blue
    ax2.set_ylabel('theta', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_dat, theta_hist, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = axs[1]

    # For Target
    color = red
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Target data', color=color)
    ax3.plot(x_d, y, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

    # For Fitted line
    color = blue
    ax4.set_ylabel('Fitted line', color=color)  # we already handled the x-label with ax1
    ax4.plot(x_d, np.dot(theta, x_d), color=color)
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.suptitle('Theta vs')
    plt.show()


def generate_plot_gif(hist, y):
    # Initialize plotting layout
    fig, axs = plt.subplots(ncols=2, nrows=1)
    x_d = range(len(y))
    error_hist = [d[0] for d in hist]
    theta_hist = [d[1] for d in hist]

    error_min_max = min(error_hist), max(error_hist)
    theta_min_max = min(theta_hist), max(theta_hist)
    x = []
    e, d = [], []
    def init():
        ax1 = axs[0]

        ax1.set_xlim(0, len(y))
        ax1.set_ylim(*error_min_max)
        # For Errors
        color = red
        ax1.set_xlabel('Data Index')
        ax1.set_ylabel('Error value', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        *_, l1 = ax1.plot([], [], 'ro')

        ax2 = ax1.twinx()
        ax2.set_ylim(*theta_min_max)
        *_, l2 = ax2.plot([], [], 'm*')

        generate_plot_gif.l1 = l1
        generate_plot_gif.l2 = l2

        # For theta
        color = blue
        ax2.set_ylabel('theta', color=color) 
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = axs[1]

        # For Target
        color = red
        ax3.set_xlabel('Data Index')
        ax3.set_ylabel('Target data', color=color)
        ax3.plot(x_d, y, color=color)
        ax3.tick_params(axis='y', labelcolor=color)

        ax4 = ax3.twinx()
        generate_plot_gif.ax4 = ax4

        # For Fitted line
        color = blue
        ax4.set_ylabel('Fitted line', color=color)  # we already handled the x-label with ax1
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.set_ylim(0, 4000)


    def update(data):
        update.counter+=1
        error, theta = data

        e.append(error)
        d.append(theta)
        x.append(update.counter)

        generate_plot_gif.l1.set_data(x, e)
        generate_plot_gif.l2.set_data(x, d)

        generate_plot_gif.ax4.clear()
        plot_dat = np.dot(theta, x_d)
        generate_plot_gif.ax4.plot(plot_dat, color=blue)
        # generate_plot_gif.ax4.plot((plot_dat - np.mean(plot_dat))/np.std(plot_dat), color=blue)

    update.counter = 0
    ani = FuncAnimation(fig, update, iter(hist), init_func=init)
    fig.tight_layout()
    fig.suptitle('Gradient Descent Experiment')
    writer = PillowWriter(fps=25)
    ani.save("demo_grad.gif", writer=writer)