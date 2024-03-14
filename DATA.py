import random

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import McGlass


# ==============================================================================#

# Method for resolving differential equations in order to write the trajectory
@njit
def RK4(n, x0, h_f, t_max_f, params_f):
    # Rk4 method with parameters:
    # n->Dimensionality of the system
    # x0->Initial condition
    # h_f->Time step
    # t_max_f->Maximum time for the simulation
    # params_f->Parameters for the function of evaluation

    # Calculate the dimension of the time vector. It must be an integer
    t_dim = int(t_max_f / h_f) + 1

    # Calculate the vector of times. It is a vector of linearly spaced steps with size h from 0 to T
    t = np.linspace(0, t_max_f, t_dim)

    # create the vector of the solution. It has size n (dimension of the system) times T,the number of
    # times where we will evaluate the solution. We initialize it with zeros.
    r_pos = np.zeros((n, t_dim))

    r_pos[:, 0] = x0  # initialize the system
    h2 = h_f / 2  # half the step
    h6 = h_f / 6  # a sixth of the step

    # loop on times, from 1 to T (the time 0 has already been set)
    for tt in range(1, t_dim):
        k1 = func(r_pos[:, tt - 1], t[tt - 1], params_f)
        k2 = func(r_pos[:, tt - 1] + h2 * k1, t[tt - 1] + h2, params_f)
        k3 = func(r_pos[:, tt - 1] + h2 * k2, t[tt - 1] + h2, params_f)
        k4 = func(r_pos[:, tt - 1] + h_f * k3, t[tt - 1] + h_f, params_f)
        r_pos[:, tt] = r_pos[:, tt - 1] + h6 * (k1 + 2 * k2 + 2 * k3 + k4)  # update the vector according to the method

    return t, r_pos


# Here we write the equation of the "movement" that describes our system
# Lorenz equations in this case

# PARAMETERS
sigma = 10  # Parameters for the Lorenz equations
b_lorenz = 2.67  # Parameters for the Lorenz equations
r = 28  # Parameters for the Lorenz equations

a = 0.2  # Parameters for the Rossler equations
b_rossler = 0.2  # Parameters for the Rossler equations
c = 5.7  # Parameters for the Rossler equations


# In order to choose a function, change the "select_fun" parameter
@njit
def func(r_pos, t, params):
    # Function f(r_pos(t),t). It takes as arguments:
    # r_pos -> The value of (X,Y,Z) at the particular time
    selec_fun = 2

    # Lorenz Attractor
    if selec_fun == 1:
        x, y, z = r_pos
        return np.array([sigma * (y - x), x * (r - z) - y, x * y - b_lorenz * z])
    # Rossler Attractor
    if selec_fun == 2:
        x, y, z = r_pos
        return np.array([-y - z, x + a * y, b_rossler + z * (x - c)])

def plot_3Dtray(r0, r_tray):
    origen = np.array([0, 0, 0])
    c_mas = np.array([np.sqrt(b_lorenz * (r - 1)), np.sqrt(b_lorenz * (r - 1)), r - 1])
    c_men = np.array([-np.sqrt(b_lorenz * (r - 1)), -np.sqrt(b_lorenz * (r - 1)), r - 1])

    # Plot some trajectories:--------------------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.))  # The "1." indicates de relation between the length and width
    fig.suptitle('Lorenz Attractor')

    # First subplot
    ax = fig.add_subplot(2, 2, 1, projection='3d')  # (2,2) matrix, position 1
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.plot(*r_tray, lw=0.5, color="darkblue", label="Forwards trajectory")  # Plotting the trajectory
    ax.plot(*r0, "o", ms=5, color="red", label="Initial condition")  # Plotting  Initial Condition

    ax.plot(*origen, "o", ms=5, color="green", label="Point (0,0,0)")  # Plotting  Initial Condition
    ax.plot(*c_mas, "o", ms=5, color="deeppink", label="Point $C_{+}$")  # Plotting  Initial Condition
    ax.plot(*c_men, "o", ms=5, color="purple", label="Point $C_{-}$")  # Plotting  Initial Condition

    # Second subplot
    ax = fig.add_subplot(2, 2, 2)  # (2,2) matrix, position 2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    ax.plot(r_tray[0], r_tray[1], lw=0.5, color="darkblue")  # Plotting the trajectory
    ax.plot(r0[0], r0[1], "o", ms=5, color="red")  # Plotting  Initial Condition

    ax.plot(origen[0], origen[1], "o", ms=5, color="green")  # Plotting  Initial Condition
    ax.plot(c_mas[0], c_mas[1], "o", ms=5, color="deeppink")  # Plotting  Initial Condition
    ax.plot(c_men[0], c_men[1], "o", ms=5, color="purple")  # Plotting  Initial Condition

    # Third subplot
    ax = fig.add_subplot(2, 2, 3)  # (2,2) matrix, position 2
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Z Axis")

    ax.plot(r_tray[0], r_tray[2], lw=0.5, color="darkblue")  # Plotting the trajectory
    ax.plot(r0[0], r0[2], "o", ms=5, color="red")  # Plotting  Initial Condition

    ax.plot(origen[0], origen[2], "o", ms=5, color="green")  # Plotting  Initial Condition
    ax.plot(c_mas[0], c_mas[2], "o", ms=5, color="deeppink")  # Plotting  Initial Condition
    ax.plot(c_men[0], c_men[2], "o", ms=5, color="purple")  # Plotting  Initial Condition

    # Fourth subplot
    ax = fig.add_subplot(2, 2, 4)  # (2,2) matrix, position 2
    ax.set_xlabel("Y Axis")
    ax.set_ylabel("Z Axis")

    ax.plot(r_tray[1], r_tray[2], lw=0.5, color="darkblue")  # Plotting the trajectory
    ax.plot(r0[1], r0[2], "o", ms=5, color="red")  # Plotting  Initial Condition

    ax.plot(origen[1], origen[2], "o", ms=5, color="green")  # Plotting  Initial Condition
    ax.plot(c_mas[1], c_mas[2], "o", ms=5, color="deeppink")  # Plotting  Initial Condition
    ax.plot(c_men[1], c_men[2], "o", ms=5, color="purple")  # Plotting  Initial Condition

    fig.legend(loc='upper left', fontsize=15)
    # fig.tight_layout()
    plt.show()  # Showing the plot

def plot_axis_vs_time(r0, t, r_tray):
    origen = np.array([0, 0, 0])
    c_mas = np.array([np.sqrt(b_lorenz * (r - 1)), np.sqrt(b_lorenz * (r - 1)), r - 1])
    c_men = np.array([-np.sqrt(b_lorenz * (r - 1)), -np.sqrt(b_lorenz * (r - 1)), r - 1])

    # Plot some trajectories:--------------------------------------------------------
    fig = plt.figure(figsize=plt.figaspect(1.))  # The "1." indicates de relation between the length and width
    fig.suptitle('Lorenz Attractor')

    # First subplot
    ax = fig.add_subplot(2, 2, 1, projection='3d')  # (2,2) matrix, position 1
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.plot(*r_tray, lw=0.5, color="darkblue", label="Forwards trajectory")  # Plotting the trajectory
    ax.plot(*r0, "o", ms=5, color="red", label="Initial condition")  # Plotting  Initial Condition

    #ax.plot(*origen, "o", ms=5, color="green", label="Point (0,0,0)")  # Plotting  Initial Condition
    #ax.plot(*c_mas, "o", ms=5, color="deeppink", label="Point $C_{+}$")  # Plotting  Initial Condition
    #ax.plot(*c_men, "o", ms=5, color="purple", label="Point $C_{-}$")  # Plotting  Initial Condition

    # Second subplot
    ax = fig.add_subplot(2, 2, 2)  # (2,2) matrix, position 2
    ax.set_xlabel("time")
    ax.set_ylabel("X Axis")

    ax.plot(t, r_tray[0], lw=0.5, color="darkblue")  # Plotting the trajectory
    ax.plot(t[0], r0[0], "o", ms=5, color="red")  # Plotting  Initial Condition

    #ax.plot(origen[0], origen[1], "o", ms=5, color="green")  # Plotting  Initial Condition
    #ax.plot(c_mas[0], c_mas[1], "o", ms=5, color="deeppink")  # Plotting  Initial Condition
    #ax.plot(c_men[0], c_men[1], "o", ms=5, color="purple")  # Plotting  Initial Condition

    # Third subplot
    ax = fig.add_subplot(2, 2, 3)  # (2,2) matrix, position 2
    ax.set_xlabel("time")
    ax.set_ylabel("Y Axis")

    ax.plot(t, r_tray[1], lw=0.5, color="darkblue")  # Plotting the trajectory
    ax.plot(t[0], r0[1], "o", ms=5, color="red")  # Plotting  Initial Condition

    #ax.plot(origen[0], origen[2], "o", ms=5, color="green")  # Plotting  Initial Condition
    #ax.plot(c_mas[0], c_mas[2], "o", ms=5, color="deeppink")  # Plotting  Initial Condition
    #ax.plot(c_men[0], c_men[2], "o", ms=5, color="purple")  # Plotting  Initial Condition

    # Fourth subplot
    ax = fig.add_subplot(2, 2, 4)  # (2,2) matrix, position 2
    ax.set_xlabel("time")
    ax.set_ylabel("Z Axis")

    ax.plot(t, r_tray[2], lw=0.5, color="darkblue")  # Plotting the trajectory
    ax.plot(t[1], r0[2], "o", ms=5, color="red")  # Plotting  Initial Condition

    #ax.plot(origen[1], origen[2], "o", ms=5, color="green")  # Plotting  Initial Condition
    #ax.plot(c_mas[1], c_mas[2], "o", ms=5, color="deeppink")  # Plotting  Initial Condition
    #ax.plot(c_men[1], c_men[2], "o", ms=5, color="purple")  # Plotting  Initial Condition

    fig.legend(loc='upper left', fontsize=15)
    # fig.tight_layout()
    plt.show()  # Showing the plot

def getData(N):
    """
    Input:
        N       = number of values
    Output:
        u       = input values
        y       = function values
    """
    # Integration PARAMETERS:-----------------------------------------------------------------
    n_dim = 3  # Dimensionality of the system
    h = 0.01  # Integration step Lorenz
    h = 0.1 # Integration step Rossler
    params = 0  # Useless in this code
    t_max = 50000

    mc_glass = True
    while True:
        if not mc_glass:
            # generate input
            x_o = random.uniform(-10, 10)
            y_o = random.uniform(-10, 10)
            z_o = random.uniform(-10, 10)
            r0_f = np.array([x_o, y_o, z_o])  # Initial Condition
            
            #r0_f = np.array([1.0, 0.5, 0.9]) # Init para Lorenz
            r0_f = np.array([-1.0, 1.0,1.0]) # Init para Rossler
            
            # Calculating the trajectory
            t, r_tray = RK4(n_dim, r0_f, h, t_max, params)  

            maximo = max(np.abs(r_tray[0]))
            r_tray[0] = r_tray[0] / maximo

            # generate input
            # r_tray contain more elements than N
            # so we take N + 1000 elements from r_tray[i]
            quitar = len(r_tray[0]) - (N + 1000 + 1)
            u = r_tray[0][quitar:]

        else:
            # print("un crack el micky")
            u, t = McGlass.McGlass_data(N + 1000 + 1)
            maximo = max(np.abs(u))
            u = u / maximo

        # generate output arrays
        y = np.zeros(shape=(N + 1000))

        # calculate intermediate output
        for i in range(0, N + 1000):
            y[i] = u[i + 1]
        
        # throw away the first 1000 values (random number), since we need
        # to allow the system to "warm up"
        u = np.delete(u,N + 1000)
        u = u[1000:]
        y = y[1000:]

        # if all values of y are finite, return them with the corresponding
        # inputs
        if np.isfinite(y).all():
            return u, y
        # otherwise, try again. You random numbers were "unlucky"
        else:
            print('...[*] Divergent time series. Retry...')

"""
n_dim = 3  # Dimensionality of the system
h = 0.001  # Integration step
params = 0  # Useless in this code
t_max = 150

# generate input
x_o = random.uniform(-10, 10)
y_o = random.uniform(-10, 10)
z_o = random.uniform(-10, 10)
r0_f = np.array([x_o, y_o, z_o])  # Initial Condition

# Calculating the trajectory
t, r_tray = RK4(n_dim, r0_f, h, t_max, params)  

plot_axis_vs_time(r0_f, t, r_tray)
"""