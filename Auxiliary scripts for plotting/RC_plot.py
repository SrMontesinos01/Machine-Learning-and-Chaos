import numpy as np
import matplotlib.pyplot as plt


dist_m = np.load("dist_matrix.npy")
tau = (20.0/100.0) * np.mean(dist_m)
print(tau)

R = np.load("R.npy")
   
fig, ax = plt.subplots()
plt.ylim(0,4000)
im = ax.imshow(R, cmap = "hot")
cbar = ax.figure.colorbar(im, ax = ax)
cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
plt.show()