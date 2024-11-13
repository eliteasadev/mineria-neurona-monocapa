import numpy as np
import pandas as pd

CSV_PATH = "./data/monocapa.csv"

def aplicar_funcion_escalon(matriz):
    """
    Convierte los varoles de ka matriz en los valores binarios.
    Los valores >= 0 se convierte en 1 y los > 0 se convierte en 0.
    :param matriz:
    :return: Matiz transformada con valores binarios.
    """
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i][j] >= 0:
                matriz[i][j] = 1
            else :
                matriz[i][j] = 0
    return matriz

# Gargar los datos desde un archivo CSV
dataframe = pd.read_csv(CSV_PATH)
entradas_rgb = dataframe[['P1', 'P2', 'P3']].to_numpy()
salidas_esperadas = dataframe[['a1', 'a2', 'a3']].to_numpy()

# Inicializar pesos y bias
pesos = np.random.rand(3,3) - 1
bias = np.random.rand(1,3) - 1
errores = np.zeros((entradas_rgb.shape[0],3))

# Entrenamiento del modelo
for epoca in range(500):
    for indice in range(entradas_rgb.shape[0]):
        salida_actual = aplicar_funcion_escalon(np.dot(pesos, entradas_rgb[indice].T) + bias)
        errores[indice] = salidas_esperadas[indice] - salida_actual
        pesos += errores[indice] * entradas_rgb[indice] * 0.01
        bias += errores[indice]

# Imprimir resultados finales
print("Pesos finales:", pesos)
print("Bias finales:", bias)
print("Errores acumulados:", errores)
print("------------------------------------")

# Evaluar el modelo con las entradas originales
for indice in range(len(entradas_rgb)):
    salida_final = aplicar_funcion_escalon(np.dot(pesos, entradas_rgb[indice].T) + bias)
    print(f"Salida para entrada {indice}: {salida_final}")
    print(f"El valor esperado coincide con la salida obtenida:", salidas_esperadas[indice] == salida_final)
    print("------------------")
