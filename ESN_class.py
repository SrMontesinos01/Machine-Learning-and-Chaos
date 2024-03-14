# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:53:42 2023

@author: Daniel
"""
import numpy as np
import random
from numba import njit
import matplotlib.pyplot as plt


# Método para regla de plasticidad antiHeb
@njit(nopython = True)
def plastic_antiHeb(eta, W, x, x_ant):
    W_fin = np.copy(W)
    n = len(W[0])
    x_ant_eta = x_ant * eta
    
    for k in range(n):
        x_k = x[k]  # Almacenamos x[k] para evitar el acceso repetido
        sum_val = np.sum((W[k, :] - x_k * x_ant_eta) ** 2)
        sum_val = np.sqrt(sum_val)
        
        W_fin[k, :] = (W[k, :] - x_k * x_ant_eta) / sum_val
    
    # delta = np.sum(W_fin - W)
    # print(delta)
    return W_fin


# Métodos para el ruido en las activaciones
def sigmoid(x):
  
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig

def calc_sigma(x, delta, p, c):
    num = sigmoid(p * delta)
    value = - np.sign(x) * c * (num - 0.5)**2
    
    return value

class ESN:
    
    # Método constructor para inicializar la ESN
    def __init__(self, inSize, outSize, resSize, alpha, sparsity, sp_rad, 
                 scaling, l_n):
        self.sp_rad = sp_rad
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize
        self.alpha = alpha
        self.sparsity = sparsity
        self.scaling = scaling
        
        self.x = np.random.uniform(-self.scaling,self.scaling, 
                                   size=(self.resSize))
        
        self.eff_input = [] # Para guardar el input efectivo
        self.eta = 10 ** (-6) # Parámetro para reglas de plasticidad
        self.noise_p = random.uniform(-1, 1) # Parámetro p del ruido pro
        self.l_noise = l_n # Ruido que multiplica la Gaussiana o lo que sea
        
        # Calculamos la matriz Win
        # +1 for Bias
        self.Win = np.zeros((resSize, inSize + 1))
        for i in range(resSize):
            for j in range(inSize + 1):
                self.Win[i][j] = random.uniform(-1, 1)
                
        # Se define la matriz W teniendo en cuenta el sparsity
        # Si sparsity = 1 entonces toda la matriz es nula
        # Si sparsity = 0 entonces están todas las conexiones
        self.W = np.zeros((resSize, resSize))
        for i in range(resSize):
            for j in range(resSize):
                prob = random.random()
                if prob < self.sparsity:
                    self.W[i][j] = 0
                else:
                    self.W[i][j] = random.uniform(-1, 1)
        
        # Ajustamos el radio espectral
        spec_rad =  max(abs(np.linalg.eigvals(self.W)))
        self.W = self.W * (sp_rad  / spec_rad) 
        self.sp_rad = max(abs(np.linalg.eigvals(self.W)))
    
    # ----------------------- Activaciones Neuronales ----------------------- #
    # Método para calular las activaciones a partir de un input
    def calcular_activaciones(self, u, antiHeb, int_noise, ext_noise, 
                              ext_noise_p,):
        x_ant = self.x  # Se guarda el valor de las activaciones para AntiHeb
        
        u_1 = np.hstack((1,u))
        A = np.dot(self.Win, u_1)
        B = np.dot(self.W, self.x)
        C = self.scaling * A + B
        
        # Sea aplica el ruido dentro de la función de activación
        if int_noise:
            num = C + self.l_noise * np.random.normal(0, 1, len(C))
            x_tilde = np.tanh(num)
            print(self.l_noise, " int")
            
        else:
            x_tilde = np.tanh(C)
            
        # Sea aplica el ruido fuera de la función de activación
        if ext_noise:
            self.x = (1 - self.alpha) * self.x + self.alpha * x_tilde
            self.x = self.x + self.l_noise * np.random.normal(0, 1, len(C))
            # print(self.l_noise, " ext1")
            
        elif ext_noise_p:
            delta = x_tilde - C
            epsilon = np.abs(np.random.normal(0, 1, len(C)))
            
            sigma = calc_sigma(C, delta, self.noise_p, self.l_noise)
            
            self.x = (1 - self.alpha) * self.x + self.alpha * x_tilde
            self.x = self.x + sigma * epsilon
            
            # print (np.mean(sigma * epsilon))
            
        else:
            self.x = (1 - self.alpha) * self.x + self.alpha * x_tilde
        
        # Se guardan el input efectivo en una lista de la clase
        self.eff_input.append(A)
            
        if antiHeb:
            # Se utiliza la regla de plasticidad antiHeb
            self.W  = plastic_antiHeb(self.eta, self.W, self.x, x_ant)
            self.sp_rad = max(abs(np.linalg.eigvals(self.W)))
        
        return np.append(np.append(1, u), self.x)
        
    # ----------------------- PLOTEAR Y GUARDAR COSAS ----------------------- #
    # Plotea un histograma de las activaciones neuronales
    def hist_act(self):
        plt.hist(self.x, 15)
        
    # Guarda el Input efectivo
    def save_eff_Input(self):
        np.save("eff_input.npy", self.eff_input)