# -*- coding: utf-8 -*-
"""
Created on Sat May 27 18:53:17 2023

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

import DATA
import ESN_class_noise
import warnings

# Filtrar todas las advertencias
# A veces sale warning de la función de calcular los coeficientes de Lyapunov
# El warning está relacionado con "nan" o divisones por cero
# Al final da igual ya que esos valores se ignoran al pasar el resutlado
warnings.filterwarnings("ignore")

# ------------------- Functions for Lyapunov Exponents -------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Médtodo para calcular activaciones en el cálculo del exponenete de Lya
def calcular_activaciones_lyapunov(u, activaciones, network): 
         
    u_1 = np.hstack((1,u))
    A = np.dot(network.Win, u_1)
    B = np.dot(network.W, activaciones)
    x_tilde = np.tanh(network.scaling * A + B)
    act = (1 - network.alpha) * activaciones + network.alpha * x_tilde
    
    return np.append(np.append(1, u), act)

# Función para calcular el exponenete
def lyap_exponent(Tmax, transient, X, network, Wout):
    
    N_pert = 10
    gamma0 = 1e-12
    lyapn = np.zeros(N_pert)
    R_1 = X.copy()
    
    for n_perturb in range(N_pert):
        gammak = np.zeros(Tmax-transient)
        # x = np.zeros((N,1)) # Reservoir states (T, N)    
        
        for t in range(transient):
            y_1 = np.dot(R_1[2:],Wout)
            u_1 = y_1
            R_1 = calcular_activaciones_lyapunov(u_1, R_1[2:], network)
            
        R_2 = R_1.copy()
        R_2[n_perturb] = R_2[n_perturb] + gamma0  
        for t in range(transient, Tmax):
            y_1 = np.dot(R_1[2:],Wout)
            u_1 = y_1
            
            R_1 = calcular_activaciones_lyapunov(u_1,R_1[2:], network)
            R_2 = calcular_activaciones_lyapunov(u_1,R_2[2:], network)
                         
            gammak[t-transient] = np.linalg.norm(R_1 - R_2)
            R_2 = R_1 + (gamma0/gammak[t-transient])*(R_2 - R_1)
        
        
        lyapn[n_perturb] = np.mean(np.log(gammak/gamma0))
        mean = np.mean(lyapn[~np.isnan(lyapn)])
        std = np.std(lyapn[~np.isnan(lyapn)])
        
    return mean, std

# ------------------- Aux. Functions the TEST ----------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Devuelve la Media y Varianza de una lista de datos
def calc_media_y_var(list1):
    media = np.mean(list1)
    var = np.std(list1)
    return media, var
    
# Función para hacer la regresión lineal usando la regresión Ridge
def ridge_regression(target, x, beta):
        # A = np.dot(np.transpose(x),target)  # Dim (resSize + 2)x(1)
        A = np.dot(target, np.transpose(x))   # Dim (resSize + 2)x(1)
        B = np.dot(x,np.transpose(x))      # Dim (resSize + 2)x(resSize + 2)
        iden = np.identity(len(B[0]))       
        C = np.linalg.inv(B + beta * iden)
        D = np.dot(A,C) # Dim res
        return D

# Función que usa la ESN y hace todo
def test(inSize = 1, outSize = 1, train_cycles = 4000, test_cycles = 1000, 
             alpha = 1, sp_r = 0.9, spar = 0.9, scal = 0.8, resSize = 300, 
             epochs = 1, l_n = 1,int_n = False, ext_n = False, ext_np = False,
             antiH = False, antiO = False, PI = False):
    
    # Primero obtenemos los datos usando el fichero DATA.py
    data, y = DATA.getData(train_cycles + test_cycles)
    data_train, y_train = data[:train_cycles], y[:train_cycles]
    data_test, y_test = data[train_cycles:], y[train_cycles:]
    
    # Creamos la ESN
    network = ESN_class_noise.ESN(inSize, outSize, resSize, alpha, spar,sp_r,
                                  scal, l_n)
    fig = plt.plot()
    plt.xlabel("Tiempo", fontsize = 20)
    plt.ylabel("X(t) Lorenz", fontsize = 20)
    plt.plot(data_train)
    #---------------- ENTRENAMIENTO ----------------#
    input_eff_bool = True  # Activar para guardar matrices de input efectivo
    dm_train_list = []
    for i in range(epochs):
        dm_train = np.zeros((train_cycles, 1 + inSize + resSize))
        for i in range(0, train_cycles):
            u_n = data_train[i]
            R_train = network.calcular_activaciones(u_n, antiH, int_n,
                                                    ext_n, ext_noise_p)
            dm_train[i] = R_train # Almacenamos todas las activaciones
        
        if input_eff_bool:
            dm_train_list.append(dm_train)
    
    # Se guarda las matrices de activaciones de todo el training para la 
    # gráfica del movimiento de la activación neuronal en el espacio de fases
    # para comparar con plasticidad y sin plasticidad
    if input_eff_bool:
        network.save_eff_Input()
        np.save("serie_temporal_train.npy", data_train)
        if antiH:
            np.save("dm_train_list_Heb.npy", dm_train_list)
        else:
            np.save("dm_train_list.npy", dm_train_list)
        
    
    dm_train = np.transpose(dm_train)
    y_train = np.transpose(y_train)
    Wout = ridge_regression(y_train, dm_train[2:], 1*10**(-7))
    
    #---------------- TEST DE PREDICCIÓN ----------------#
    dm_test = np.zeros((test_cycles, 1 + inSize + resSize))
    u_n = data_test[0]
    prediction = [u_n]
    R_test = network.calcular_activaciones(u_n, False, False, False, False)
    
    # Se calcula el coeficiente de Lyapunov
    transient = 50
    Tmax = test_cycles
    lya_exp, lya_std = lyap_exponent(Tmax, transient, R_test, network, Wout)
    
    for i in range(1, test_cycles):
        y_n = np.dot(R_test[2:],Wout)
        u_n = y_n
        prediction.append(y_n)
        R_test = network.calcular_activaciones(u_n, False, False, False, False)
        dm_test[i] = R_test # Almacenamos todas las activaciones
        
    
    #---------------- NRMSE y RMSE ----------------#
    prediction = np.array(prediction)
    NRMSE = np.sqrt(np.divide(np.mean(np.square(y_test - prediction))
                              , np.var(y_test)))

    RMSE = np.sqrt(np.mean(np.square(y_test - prediction)))
    #---------------- FPP  ----------------#
    y_max = np.amax(data_test)
    y_min = np.amin(data_test)
    dif_max = (y_max - y_min) * 0.03
    
    FPP = 0
    for i in range(len(prediction)):
        dif_exp = abs(prediction[i] - data_test[i])
        if dif_exp > dif_max:
            FPP = i
            break
    
    #---------------- PLOTS Predicción ----------------#
    if True:
        
       # Finalmente se plotea Data Test con la predicción
       fig = plt.figure()
       plt.title("Valores de la Serie Temporal real frente a la predicción")
       plt.plot(data_test, label="data_test")
       plt.plot(prediction, linestyle="--", label="Predicción")
       plt.legend()
       plt.show()
       
       # Se plotea la activación neuornal
       fig = plt.figure()
       R = dm_test[:, -resSize:]  # Seleccionamos los últimos resSize elementos de cada fila
       R_input = dm_test[:, 1:-resSize]
       limit = test_cycles  # Se toman todos los ciclos del test para representar
       nr_neurons = 8  # Número de neuronas que se represetan
       plt.title("Actividad Neuronal e Input")
       plt.plot(R[:limit, :nr_neurons], linewidth=1)
       plt.plot(R_input[:limit], linestyle="--", color='k',
                label='Input Signals', linewidth=2)
       plt.legend(loc='upper right')
      
    return FPP, NRMSE, network.sp_rad, lya_exp



# ====================== CÓDIGO PRINCIPAL ====================== #
train_c = 4000
test_c = 1000
resSize = 300  # Numero de Neuronas
alpha = 1  # Pertenece a (0,1], es el leaking rate
radio = 1.4 # Radio espectral
sparcity = 0.9  # Pertence a (0,1)
scaling = 2.4
c = 1e-3

PI_plastic = False
antiOja = False
antiHeb = False

int_noise = False
ext_noise = False
ext_noise_p = False

# Variables que he usado para recorrer los bucles for
num_epoch = list(range(1,2))
repes = 1

inicio = 0.5
fin = 2.5
cantidad = int((fin - inicio) / 0.05) + 1
radios_sp = np.linspace(inicio, fin, cantidad)

inicio = 1e-5
fin = 1e-3
cantidad = int((fin - inicio) / 1e-5) + 1
c_params = np.linspace(inicio, fin, cantidad)

noise_list = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
# Usar esto de abajo para probar con una iteracón de ruido
noise_list = [0] 

# +++++++++++++++++++ Se escriben los parámetros usados +++++++++++++++++++
archivo = open("parametros_test.txt","w")
archivo.write("train_c = " + str(train_c) + "\n")
archivo.write("test_c = "+ str(test_c)+ "\n")
archivo.write("resSize = "+ str(resSize)+ "\n")
archivo.write("alpha = "+ str(alpha)+ "\n")
archivo.write("radio espectral = "+ str(radio)+ "\n")
archivo.write("sparcity = "+ str(sparcity)+ "\n")
archivo.write("scaling = "+ str(scaling)+ "\n")
archivo.write("\n")
archivo.write("antiHeb = "+ str(antiHeb)+ "\n")
archivo.write("intNoise = "+ str(int_noise)+ "\n")
archivo.write("extNoise = "+ str(ext_noise)+ "\n")
archivo.write("extNoise_pro = "+ str(ext_noise_p)+ "\n")
archivo.write("\n")
archivo.write("radios_sp = "+ str(radios_sp)+ "\n")
archivo.write("Noise_list = "+ str(noise_list)+ "\n")
archivo.close()

# Listas para hacer la estadisitca (medias y varianzas)
FPP_total_epochs = []
FPP_var_total_epochs = []

RMSE_total_epochs = []
RMSE_var_total_epochs = []

spr_total_epochs = []
spr_var_total_epochs = []

lya_total_epochs = []
lya_var_total_epochs = []



repetir = False
epoch = 1

# Este bucle se puede usar para recorrer  valores de diferentes variables
# "for radio in radios_sp" haciendo repeticiones en cada valor
# "for epoch in num_epoch" haciendo repeticiones en cada valor
noise_exponent = -8
for noise in noise_list:
    noise_exponent = noise_exponent + 1
    print("+++++++++++++++++++ Ruido",noise,"+++++++++++++++++++")
    
    FPP_total_epochs = []
    FPP_var_total_epochs = []

    RMSE_total_epochs = []
    RMSE_var_total_epochs = []

    spr_total_epochs = []
    spr_var_total_epochs = []

    lya_total_epochs = []
    lya_var_total_epochs = []
    
    for epoch in num_epoch:
        FPP_list = []
        RMSE_list = []
        spr_list = []
        lya_list = []
        #sp_r = radio
    
        print("-------------------- Radio",radio," --------------------")
        
        for i in range(repes):
            
            if i % 10 == 0:
                print(i)
            
            FPP, RMSE, spr, lya = test(train_cycles=train_c, 
                                       test_cycles=test_c, resSize=resSize, 
                                       spar=sparcity, alpha=alpha, 
                                       sp_r=radio, 
                                       scal=scaling,epochs=epoch, 
                                       antiH=antiHeb, antiO=antiOja, 
                                       PI=PI_plastic, l_n=noise, 
                                       int_n = int_noise, ext_n= ext_noise,
                                       ext_np=ext_noise_p)
            
            # Si se le va la cabeza a la red y sale un error muy grande
            # se repite la iteración
            # Según lo que se quiera hacer, es mejor no usar esto
            if RMSE < -1:
                repetir = True
                repetir = False
                print("Repitiendo iteración. RMSE:", RMSE)
                while repetir:
                    FPP, RMSE, spr, lya = test(train_cycles=train_c, 
                                               test_cycles=test_c, resSize=resSize, 
                                               spar=sparcity, alpha=alpha, 
                                               sp_r=radio, 
                                               scal=scaling,epochs=epoch, 
                                               antiH=antiHeb, antiO=antiOja, 
                                               PI=PI_plastic)
                    if RMSE < 0.7:
                        repetir = False
            
            FPP_list.append(FPP)
            RMSE_list.append(RMSE)
            spr_list.append(spr)
            lya_list.append(lya)
            
        # Se calcula la media y varianza de cada variable
        FPP_medio, FPP_var = calc_media_y_var(FPP_list)
        RMSE_medio, RMSE_var = calc_media_y_var(RMSE_list)
        spr_medio, spr_var = calc_media_y_var(spr_list)
        lya_medio, lya_var = calc_media_y_var(lya_list)
        
        # Se muestran las medias y varianzas de cada variable
        print("")
        print("FPP_medio:" , FPP_medio , "FPP_var:", FPP_var)
        print("RMSE_medio:" , RMSE_medio , "RMSE_var:", RMSE_var)
        print("spr_medio:" , spr_medio , "spr_var:", spr_var)
        print("lya_medio:" , lya_medio , "lya_var:",lya_var)
        print("")
        
        # Hcemos una lista que tenga todos los valores medios y varianzas
        FPP_total_epochs.append(FPP_medio)
        FPP_var_total_epochs.append(FPP_var)
        
        RMSE_total_epochs.append(RMSE_medio)
        RMSE_var_total_epochs.append(RMSE_var)
        
        spr_total_epochs.append(spr_medio)
        spr_var_total_epochs.append(spr_var)
        
        lya_total_epochs.append(lya_medio)
        lya_var_total_epochs.append(lya_var)
    
        # Guardamos los datos con sus erorres en archivos
        FPP_y_var = list(zip(FPP_total_epochs, FPP_var_total_epochs))
        ruta_archivo = 'FPP_y_var' + str(noise_exponent) +'.csv'
        with open(ruta_archivo, 'w', newline='') as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerows(FPP_y_var)
        
        RMSE_y_var = list(zip(RMSE_total_epochs, RMSE_var_total_epochs))
        ruta_archivo = 'RMSE_y_var' + str(noise_exponent) +'.csv'
        with open(ruta_archivo, 'w', newline='') as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerows(RMSE_y_var)
        
        spr_y_var = list(zip(spr_total_epochs, spr_var_total_epochs))
        ruta_archivo = 'spr_y_var.csv'
        with open(ruta_archivo, 'w', newline='') as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerows(spr_y_var)
        
        lya_y_var = list(zip(lya_total_epochs, lya_var_total_epochs))
        ruta_archivo = 'lya_y_var' + str(noise_exponent) +'.csv'
        with open(ruta_archivo, 'w', newline='') as archivo:
            escritor_csv = csv.writer(archivo)
            escritor_csv.writerows(lya_y_var)


