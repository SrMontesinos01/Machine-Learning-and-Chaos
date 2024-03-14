import numpy as np
import matplotlib.pyplot as plt



def calcular_dist(train, R):
    dist_matrix = np.zeros((dm_train.shape[1],dm_train.shape[1]))
    for i in range(R.shape[0]):
        if i % 500 == 0: 
            print(i)
        for j in range(R.shape[0]):
            
            act_i = train[0, i,2:]
            act_j = train[0, j,2:]
            
            dist_matrix[i,j] = sum(abs(act_i - act_j))
    np.save("dist_matrix.npy", dist_matrix)
    return dist_matrix

def calcular_R(R, dist_m,tau):
    for i in range(R.shape[0]):
        for j in range(R.shape[0]):
            R[i,j] = np.heaviside(tau - dist_m[i,j],0)
    np.save("R.npy", R) 
    
dm_train = np.load("dm_train_list.npy")  
R = np.zeros((dm_train.shape[1],dm_train.shape[1]))

dist_m = calcular_dist(dm_train, R)
tau = (20.0/100.0) * np.mean(dist_m)
print(tau)

calcular_R(R,dist_m,tau)
       
fig, ax = plt.subplots()
plt.ylim(0,4000)
im = ax.imshow(R, cmap = "hot")
cbar = ax.figure.colorbar(im, ax = ax)
cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
plt.show()