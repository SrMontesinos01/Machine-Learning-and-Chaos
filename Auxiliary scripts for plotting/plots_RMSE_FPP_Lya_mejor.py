import matplotlib.pyplot as plt
import csv


color2 = "#87CEEB"
color1 = "orange"
color3 = "limegreen"

cap_style1 = {
    'linewidth': 1.5,  # Ancho de línea de las barras de error
    "ls": "--",
    'color': 'orange',    # Color de las barras de error
    'capsize': 6,       # Tamaño de las marcas finales de las barras de error
    "markersize": 5,
    "ecolor": "coral",
    "mec": 'coral'
}

cap_style2 = {
    'linewidth': 1.5,  # Ancho de línea de las barras de error
    "ls": "--",
    'color': '#87CEEB',    # Color de las barras de error
    'capsize': 6,       # Tamaño de las marcas finales de las barras de error
    "markersize": 5,
    "ecolor": '#4169E1',
    "mec": '#4169E1'
}

cap_style3 = {
    'linewidth': 1.5,  # Ancho de línea de las barras de error
    "ls": "--",
    'color': 'limegreen',    # Color de las barras de error
    'capsize': 6,       # Tamaño de las marcas finales de las barras de error
    "markersize": 5,
    "ecolor": 'green',
    "mec": 'green'
}

cap_style4 = {
    'linewidth': 1.5,  # Ancho de línea de las barras de error
    "ls": "--",
    'color': 'royalblue',    # Color de las barras de error
    'capsize': 6,       # Tamaño de las marcas finales de las barras de error
    "markersize": 5,
    "ecolor": 'green',
    "mec": 'green'
}

def leer_fichero(ruta):
    # Lee los datos desde el archivo CSV
    with open(ruta, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        datos_con_errores = list(lector_csv)

    # Separa los datos y errores en listas separadas
    datos_experimentales, errores_experimentales = zip(*datos_con_errores)

    # Convierte los datos de texto a números
    datos = list(map(float, datos_experimentales))
    error = list(map(float, errores_experimentales))
    
    return datos, error
# ----------------------- Lectura Fich ----------------------------------------
# ***** FPP *****
ruta_archivo = 'FPP_y_var-1.csv'


FPP1, FPP1_var = leer_fichero('FPP_y_var-1.csv')
FPP2, FPP2_var = leer_fichero('FPP_y_var-2.csv')
FPP3, FPP3_var = leer_fichero('FPP_y_var-3.csv')
FPP4, FPP4_var = leer_fichero('FPP_y_var-4.csv')
FPP5, FPP5_var = leer_fichero('FPP_y_var-5.csv')
FPP6, FPP6_var = leer_fichero('FPP_y_var-6.csv')
FPP7, FPP7_var = leer_fichero('FPP_y_var-7.csv')
spr, spr_var = leer_fichero('spr_y_var.csv')


# ----------------------- GRÁFICA FPP y RMSE ----------------------------------
# Grafica los puntos de datos con sus barras de error
plt.errorbar(spr, FPP1, yerr=FPP1_var, 
            **cap_style1 , fmt='o',label = "no noise")
plt.errorbar(spr, FPP2, yerr=FPP2_var, 
            **cap_style2 , fmt='o',label = "1e-5")
plt.errorbar(spr, FPP3, yerr=FPP3_var, 
            **cap_style3 , fmt='o',label = "1e-6")
plt.errorbar(spr, FPP4, yerr=FPP4_var, 
            **cap_style4 , fmt='o',label = "1e-4")

plt.xlabel('Radio Espectral', fontsize = 20)
plt.ylabel('FPP', color = color1, fontsize = 20)
plt.legend()
plt.show


