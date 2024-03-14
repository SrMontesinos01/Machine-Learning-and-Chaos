#Pretendo hacer una serie temporal de mackey-glass
from matplotlib import pyplot as plt
import numpy as np
import random

def mackeyglass_eq(x_t, x_t_minus_tau, a, b, c):
    return (-b*x_t + a*x_t_minus_tau/(1. + x_t_minus_tau**c))

def mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b, c):
    #No se necesita paso porque no hay dependencia explícita del tiempo
    k1 = deltat*mackeyglass_eq(x_t,          x_t_minus_tau, a, b, c)
    k2 = deltat*mackeyglass_eq(x_t+0.5*k1,   x_t_minus_tau, a, b, c)
    k3 = deltat*mackeyglass_eq(x_t+0.5*k2,   x_t_minus_tau, a, b, c)
    k4 = deltat*mackeyglass_eq(x_t+k3,       x_t_minus_tau, a, b, c)
    return x_t + k1/6. + k2/3. + k3/3. + k4/6.


# Función externa que se usa en el fichero DATA
# N es el número de datos que queremos obtener
def McGlass_data(N):
    # Inputs
    a = 0.2  # value for a
    b = 0.1  # value for b
    c = 10
    tau = 17 # delay constant
    x0 = 0.3  # initial condition: x(t=0)=x0
    deltat = 1  # time step size (which coincides with the integration step)
    sample_n = N - 1  # total no. of samples, excluding the given initial condition

    time = 0
    index = 0
    history_length = int(tau/deltat)
    x_history = np.zeros(history_length)  #here we assume x(t)=0 for -tau <= t < 0
    x_t = x0

    X = np.zeros(sample_n+1)    #vector of all generated x samples
    T = np.zeros(sample_n+1)    #vector of time samples

    for i in range(sample_n+1):
        X[i] = x_t

        if i<tau:
            x_t_minus_tau = 0.
        else:
            x_t_minus_tau = X[i-tau]

        x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b, c)

        time = time + deltat
        T[i] = time
        x_t = x_t_plus_deltat

    forward = X[1:]
    backward = X[:-1]
    if False:
        plt.plot(T, X)
        plt.show()
        plt.plot(backward,forward)
        plt.show()

    return X,T
