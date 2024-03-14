"""
Nota:   Independientemente del número de epochs o lo que sea que se esté
        variando, lo que se lee de los ficheros es la úttima matriz dm_train
        y el último conjunto de datos de input efectivo. Es decir, lo que 
        correspondería a los últimos pasos del training que se haya hecho.
        
Uso:    Para usar el código, es necesario haber ejecutado el ESN_test con
        reglas de plasticidad y sin ellas. Esto se debe a que esta diseñado
        para plotear la actividad neuronal en ambos casos y necesita esos 
        ficheros.
"""

import numpy as np
import matplotlib.pyplot as plt

neurona = 200 # Se puede escoger cualquier neurona
train_c = 4000 # Esto debe coincidir con train_cycles de ESN_test



# Cargamos los datos del input efectivo
eff_input = np.load("eff_input.npy")
plot_eff_input = []
for i in range(len(eff_input[:, 0])):
    plot_eff_input.append(eff_input[i][neurona])

# Cargamos los datos de la activación neuronal sin plasticidad
dm_train = np.load("dm_train_list.npy")
plot_actividades = []
for i in range(dm_train.shape[0]):
    for j in range(dm_train.shape[1]):
        elemento = dm_train[i, j, neurona]
        plot_actividades.append(elemento)

ult_act = dm_train[-1,-1:]
ult_act = ult_act[:,2:]

train_acts = dm_train[-1]
medias = np.mean(train_acts[:, 2:], axis = 0)

# Cargamos los datos de la activación neuronal con plasticidad
dm_train = np.load("dm_train_list_Heb.npy")
plot_actividades_Heb = []
for i in range(dm_train.shape[0]):
    for j in range(dm_train.shape[1]):
        elemento = dm_train[i, j, neurona]
        plot_actividades_Heb.append(elemento)
        
ult_act_Heb = dm_train[-1,-1:] # Se selecciona la última fila, de la última dm
ult_act_Heb = ult_act_Heb[:,2:] # Se seleccionan todas las activaciones

primer_act_Heb = dm_train[0,-1:]
primer_act_Heb = primer_act_Heb[:,2:]

train_acts = dm_train[-1]
medias_Heb = np.mean(train_acts[:, 2:], axis = 0)

# Plots
fig = plt.figure()
plt.title("Atividad Neuronal frente a Input Efectivo")
plt.xlabel("Input Efectivo")
plt.ylabel("Atividad Neuronal")
plt.plot(plot_eff_input, plot_actividades,"o", ms = 1)
# plt.plot(plot_eff_input, plot_actividades_Heb,"o", ms = 1)
plt.show()

fig = plt.figure()
plt.title("Histograma de Actividad Neuronal sin Plasticidad")
plt.xlabel("Atividad Neuronal")
plt.hist(ult_act[0],20)
# plt.hist(medias, 20)

"""
fig = plt.figure()
plt.title("Histograma de Actividad Neuronal con Plasticidad")
plt.xlabel("Atividad Neuronal")
plt.hist(ult_act_Heb[0],20)
# plt.hist(medias_Heb,20)

fig = plt.figure()
plt.title("Histograma de Actividad Neuronal con Plasticidad")
plt.xlabel("Atividad Neuronal")
plt.hist(primer_act_Heb[0],20)
"""

#------------------- PLOT 3D de Activaiciones Neuronales ----------------------
neur1 = 95
neur2 = 78
neur3 = 245

# Cargamos los datos de la activación neuronal 
dm_train = np.load("dm_train_list.npy")
act1 = []
act2 = []
act3 = []
tiempo = []
t = 0
i = 0
for j in range(dm_train.shape[1]):
    act1.append(dm_train[i, j, neur1])
    act2.append(dm_train[i, j, neur2])
    act3.append(dm_train[i, j, neur3])
    tiempo.append(t)
    j = j + 50
    t = t + 1
# First subplot
cmhot = plt.get_cmap("hot")
c = tiempo

ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel("Neurona 1 ")
ax.set_ylabel("Neurona 2 ")
ax.set_zlabel("Neurona 3 ")
ax.set_title("Actividades neuronales de 3 neuronas durante el training")
scatter = ax.scatter(act1,act2,act3, s=50, c=c, cmap=cmhot)  
plt.colorbar(scatter)
plt.show()

