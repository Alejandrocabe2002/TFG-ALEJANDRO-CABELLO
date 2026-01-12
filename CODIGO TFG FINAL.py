# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 12:25:52 2025

@author: Alejandro
"""
# Parte de inclusión de archivos externos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import _tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


# PARTE DECLARATIVA (donde declaro todas las variables que después usaré)

# Ruta del archivo CSV
file_path = r"C:\Users\Alejandro\Desktop\Parking_Violations_Issued.csv"

# Leer el archivo CSV
buffer_lectura_csv = pd.read_csv(file_path)

# Columnas numéricas del DataSet
numeric_cols = buffer_lectura_csv.select_dtypes(include=['int64', 'float64']).columns

# Creamos una tabla resumen
resumen = list()

# Valores a eliminar de las columnas
valores_a_eliminar = [
    'Violation Legal Code',
    'Time First Observed',
    'Hydrant Violation',
    'Double Parking Violation',
    'No Standing or Stopping Violation'
]

# Valores en los que el 0 no tiene sentido en las columnas, por lo que serán transformados a NaN
valores_con_ceros_invalidos = ['Vehicle Year', 'Date First Observed', 'Vehicle Expiration Date']

# Lista de variables numéricas del dataset
variables_numericas = [
    'Summons Number', 'Vehicle Expiration Date', 'Violation Location',
    'Violation Precinct', 'Issuer Precinct', 'Issuer Code',
    'Street Code1', 'Street Code2', 'Street Code3',
    'Law Section', 'Vehicle Year', 'Feet From Curb',
    'Unregistered Vehicle?', 'Violation Code', 'Date First Observed'
]

# Lista de variables numéricas que tienen outliers
variables_numericas_outliers = [
    'Vehicle Expiration Date', 'Violation Location', 'Violation Precinct',
    'Issuer Precinct', 'Issuer Code', 'Street Code1', 'Street Code2',
    'Street Code3', 'Law Section', 'Vehicle Year', 'Feet From Curb',
    'Violation Code', 'Date First Observed'
]

# Lista de variables numéricas no relevantes que vamos a eliminar
variables_no_relevantes = [
    'Summons Number',            # Identificador único
    'Law Section',               # Valor constante
    'Unregistered Vehicle?',     # Demasiados nulos / ceros
    'Feet From Curb'             # Todo son ceros
]

# Lista de variables cualitativas que analizaremos
variables_cualitativas = [
    'Registration State',
    'Plate Type',
    'Vehicle Make',
    'Vehicle Body Type',
    'Violation Description',
    'Violation County'
]

# Variables redundantes altamente relacionadas
variables_redundantes = ['Issuer Precinct', 'Violation Location'] 

# Variables del modelo
columnas_modelo = [
    'Violation Precinct', 'Issuer Code', 'Street Code1',
    'Street Code2', 'Street Code3', 'Vehicle Year',
    'Vehicle Expiration Date', 'Violation Description'
]

# Variables relevantes para el cluster
selected_columns = [
    'Violation Code', 'Vehicle Body Type', 'Vehicle Make',
    'Vehicle Year', 'Issue Date', 'Days Parking In Effect',
    'From Hours In Effect', 'To Hours In Effect', 'Street Code1',
    'Street Code2', 'Issuing Agency'
]

# Variables categóricas
label_cols = ['Vehicle Body Type', 'Vehicle Make', 'Issuing Agency',
              'Days Parking In Effect', 'From Hours In Effect', 'To Hours In Effect']

# Scaler
scaler = StandardScaler()

# Variables para el scaler
inertia = []
silhouette_scores = []

# GRÁFICOS

# Crear figura y eje
fig1, ax1 = plt.subplots(figsize=(12, 12))  # Puedes ajustar el tamaño si lo necesitas
ax1.axis('off')  # Ocultar ejes


#### PARTE EJECUTIVA (donde se hacen las cosas) ####

#### FASE 1 : LIMPIEZA ####

# Mostrar las primeras filas para verificar que se ha cargado correctamente
print(buffer_lectura_csv.head())

# Mostrar información básica de las columnas
print("\nTIPO DE VARIABLE POR COLUMNA:")
print(buffer_lectura_csv.dtypes)

# Número de valores nulos por columna
print("\nVALORES FALTANTES POR COLUMNA:")
print(buffer_lectura_csv.isnull().sum())

# Número de valores cero por columna (solo aplicable a columnas numéricas)
print("\nVALORES CERO POR COLUMNA (solo numéricas):")

# Seleccionamos solo las columnas numéricas
for col in numeric_cols:
    zero_count = (buffer_lectura_csv[col] == 0).sum()
    print(f"{col}: {zero_count}")

# Para cada columna añadimos el dato, el tipo, el incremento si no es nulo, y la suma de si es cero
for col in buffer_lectura_csv.columns:
    tipo = buffer_lectura_csv[col].dtype
    nulos = buffer_lectura_csv[col].isnull().sum()
    ceros = (buffer_lectura_csv[col] == 0).sum() if col in numeric_cols else "N/A"
    resumen.append([col, tipo, nulos, ceros])

# Creamos dataFrame para la tabla resumen y mostramos en forma de tabla
resumen_buffer_lectura_csv = pd.DataFrame(resumen, columns=['Columna', 'Tipo de dato', 'Valores faltantes', 'Valores cero'])
print("\nRESUMEN EN FORMA DE TABLA:")
print(resumen_buffer_lectura_csv.to_string(index=False))

# Crear la tabla visual
tabla = ax1.table(
    cellText=resumen_buffer_lectura_csv.values,
    colLabels=resumen_buffer_lectura_csv.columns,
    cellLoc='center',
    loc='center'
)

# Ajustar texto y tamaño de celdas de la tabla
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1.2, 1.2)

# Mostramos el título del gráfico
plt.title("Resumen del conjunto de datos, tipos de datos, valores faltantes y valores cero", fontsize=14, weight='bold')

# Lo mostrarmos en la pestaña de gráficos de Spyder
plt.show()

buffer_lectura_csv.shape[0]

# Eliminamos esas columnas del  DataSet
buffer_lectura_csv = buffer_lectura_csv.drop(columns=valores_a_eliminar)

# Confirmamos que se han eliminado
print("Columnas eliminadas:")
print(valores_a_eliminar)
print("\nColumnas actuales en el DataFrame:")
print(buffer_lectura_csv.columns)

# Reemplazar los ceros por NaN en las columnas donde 0 no tiene sentido real
for col in valores_con_ceros_invalidos:
    buffer_lectura_csv[col] = buffer_lectura_csv[col].replace(0, np.nan)
    
# Confirmar que el tratamiento se aplicó correctamente (opcional)
print("\nVerificación de tratamiento aplicado (primeras filas):")
print(buffer_lectura_csv[['Vehicle Year', 'Date First Observed', 'Vehicle Expiration Date', 'Feet From Curb', 'Unregistered Vehicle?']].head())

# Crear boxplots para detectar outliers
for var in variables_numericas:
    if var in buffer_lectura_csv.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=buffer_lectura_csv[var])
        plt.title(f'Boxplot de {var}')
        plt.xlabel(var)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

fig1, axes1 = plt.subplots(nrows=len(variables_numericas), figsize=(10, 4 * len(variables_numericas)))

for i, var in enumerate(variables_numericas):
    if var in buffer_lectura_csv.columns:
        sns.boxplot(x=buffer_lectura_csv[var], ax=axes1[i])
        axes1[i].set_title(f'Boxplot de {var}')
        axes1[i].set_xlabel('')
        axes1[i].grid(True)

plt.tight_layout()
plt.show()

# Eliminamos outliers por método IQR para cada variable numérica
for col in variables_numericas_outliers:
    if col in buffer_lectura_csv.columns:
        Q1 = buffer_lectura_csv[col].quantile(0.25)
        Q3 = buffer_lectura_csv[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        buffer_lectura_csv = buffer_lectura_csv[(buffer_lectura_csv[col] >= limite_inferior) & (buffer_lectura_csv[col] <= limite_superior)]

# Confirmamos las nuevas dimensiones del DataFrame
print("\nDimensiones después de eliminar outliers:")
print(buffer_lectura_csv.shape)

# Guardar el DataFrame limpio en un nuevo archivo CSV
buffer_lectura_csv.to_csv("dataset_limpio.csv", index=False)
print("Archivo 'dataset_limpio.csv' guardado correctamente.")

# Ubicación actual en spyder del nuevo archivo.
print(os.getcwd())

#### FASE 2 : DESCRIPCIÓN DE VARIABLES ####

# Generar resumen estadístico de variables numéricas
resumen_estadistico = buffer_lectura_csv.describe().transpose()

# Renombrar columnas para mayor claridad
resumen_estadistico = resumen_estadistico.rename(columns={
    'count': 'Número de valores',
    'mean': 'Media',
    'std': 'Desviación estándar',
    'min': 'Mínimo',
    '25%': 'Percentil 25',
    '50%': 'Mediana',
    '75%': 'Percentil 75',
    'max': 'Máximo'
})

# Mostrar la tabla resumen
print("\nRESUMEN ESTADÍSTICO DE VARIABLES NUMÉRICAS:")
print(resumen_estadistico)

# (Opcional) Mostrar en tabla visual con matplotlib
fig2, ax2 = plt.subplots(figsize=(14, 10))
ax2.axis('off')
tabla = ax2.table(
    cellText=resumen_estadistico.round(2).values,
    colLabels=resumen_estadistico.columns,
    rowLabels=resumen_estadistico.index,
    loc='center',
    cellLoc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(8)
tabla.scale(1.2, 1.2)
plt.title("Resumen Estadístico de Variables Numéricas", fontsize=14, weight='bold')
plt.show()

# Eliminamos esas variables del DataFrame
buffer_lectura_csv = buffer_lectura_csv.drop(columns=variables_no_relevantes)

# Confirmamos la eliminación
print("Variables eliminadas del análisis estadístico:")
print(variables_no_relevantes)

# Guardamos una nueva versión del dataset ya refinado
buffer_lectura_csv.to_csv("dataset_limpio_final.csv", index=False)
print("Archivo 'dataset_limpio_final.csv' guardado correctamente.")

# Para cada variable cualitativa: mostrar frecuencias y generar gráfico de barras
for var in variables_cualitativas:
    if var in buffer_lectura_csv.columns:
        print(f"\nFrecuencias absolutas de {var}:")
        print(buffer_lectura_csv[var].value_counts().head(10))  # Top 10 categorías más comunes

        print(f"\nFrecuencias relativas de {var}:")
        print((buffer_lectura_csv[var].value_counts(normalize=True).head(10) * 100).round(2))

        # Gráfico de barras
        plt.figure(figsize=(10, 5))
        buffer_lectura_csv[var].value_counts().head(10).plot(kind='bar', color='skyblue')
        plt.title(f'Top 10 categorías más frecuentes en "{var}"')
        plt.xlabel(var)
        plt.ylabel('Frecuencia absoluta')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


# Cargar el dataset limpio final (ajusta la ruta si lo guardaste en otro sitio)
buffer_lectura_csv = pd.read_csv("dataset_limpio_final.csv")

# Seleccionar solo las variables numéricas
buffer_lectura_csv_numericas = buffer_lectura_csv.select_dtypes(include=['int64', 'float64'])

# Calcular matriz de correlación
matriz_corr = buffer_lectura_csv_numericas.corr(method='pearson')

# Visualizar mapa de calor de la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={'label': 'Coeficiente de correlación'})
plt.title("Mapa de calor: Correlación entre variables numéricas", fontsize=14)
plt.tight_layout()
plt.show()

# Eliminar las variables redundantes altamente correlacionadas
buffer_lectura_csv = buffer_lectura_csv.drop(columns=variables_redundantes)

# Confirmar eliminación
print("Variables eliminadas por alta correlación:")
print(variables_redundantes)

# Guardar nueva versión del DataFrame limpio con menos redundancia
buffer_lectura_csv.to_csv("dataset_sin_redundancia.csv", index=False)
print("Archivo 'dataset_sin_redundancia.csv' guardado correctamente.")

#para saber la ruta del csv bueno
print(os.path.abspath("dataset_sin_redundancia.csv"))

#### FASE 3 : MODELO RANDOM FOREST ####

# Cargar dataset limpio
buffer_modelo_csv = pd.read_csv(r"c:\users\alejandro\.spyder-py3\dataset_sin_redundancia.csv")

df_modelo = buffer_modelo_csv[columnas_modelo].dropna()

# Codificación de variable objetivo
le = LabelEncoder()
df_modelo['Violation Description'] = le.fit_transform(df_modelo['Violation Description'])

# Separar variables
X = df_modelo.drop(columns='Violation Description')
y = df_modelo['Violation Description']

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred = modelo_rf.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.tight_layout()
plt.show()

# Reporte de clasificación
reporte = classification_report(y_test, y_pred, output_dict=True)
df_reporte = pd.DataFrame(reporte).transpose()

# Visualización del reporte
plt.figure(figsize=(8, 6))
metrics = df_reporte.loc[df_reporte.index.str.isnumeric(), ['precision', 'recall', 'f1-score']]
sns.heatmap(metrics, annot=True, cmap="Blues", fmt=".2f")
plt.title("Reporte de Clasificación - Random Forest")
plt.tight_layout()
plt.show()

# Visualizar el primer árbol del Random Forest
plt.figure(figsize=(20, 10))
plot_tree(
    modelo_rf.estimators_[0],                    # Primer árbol del bosque
    feature_names=X.columns,                     # Nombres de las variables
    class_names=le.classes_,                     # Etiquetas de la variable objetivo
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Árbol de decisión individual dentro del Random Forest")
plt.tight_layout()
plt.show()

# Índice de la variable más importante
index_var = X.columns.get_loc("Street Code1")

# Buscar un árbol donde Street Code1 esté cerca del nodo raíz
for i, tree in enumerate(modelo_rf.estimators_):
    if index_var in tree.tree_.feature[:5]:  # nodo raíz y 4 nodos siguientes
        print(f"Visualizando el árbol número {i}")
        plt.figure(figsize=(25, 12))
        plot_tree(tree,
                  feature_names=X.columns,
                  class_names=le.classes_,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(f"Árbol del Random Forest donde 'Street Code1' está en los nodos principales (Árbol {i})")
        plt.tight_layout()
        plt.show()
        break

#### FASE 4 : Extraer información de los nodos del árbol más importante del Random Forest ####

# Extraemos el primer árbol del modelo Random Forest
arbol = modelo_rf.estimators_[0]
tree = arbol.tree_
feature_names = X.columns

# Obtenemos las clases originales del LabelEncoder
clases_originales = le.inverse_transform(np.arange(len(le.classes_)))

# Mostramos información de los primeros nodos (puedes cambiar el rango)
for nodo in range(5):  # por ejemplo los 5 primeros nodos
    gini = tree.impurity[nodo]
    samples = tree.n_node_samples[nodo]
    value = tree.value[nodo][0]
    clase_predicha = clases_originales[np.argmax(value)]

    print(f"Nodo: {nodo}")
    print(f"  Gini: {round(gini, 3)}")
    print(f"  Samples: {samples}")
    print(f"  Clase final: {clase_predicha}")

    # Condición de división si no es nodo hoja
    if tree.feature[nodo] != _tree.TREE_UNDEFINED:
        feature = feature_names[tree.feature[nodo]]
        threshold = tree.threshold[nodo]
        print(f"  Condición: {feature} <= {threshold:.3f}")
    else:
        print("  Nodo hoja (no tiene condición de división)")
    print("-----")
    
# Matriz de confusión del modelo forest
# Realizar predicciones
y_pred = modelo_rf.predict(X_test)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualización
plt.figure(figsize=(12, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.tight_layout()
plt.show()

#### FASE 5 : CLUSTER ####
# Cargar tu dataset
buffer_cluster_csv = pd.read_csv(r"c:\users\alejandro\.spyder-py3\dataset_sin_redundancia.csv")

# Limpiar nombres de columnas
buffer_cluster_csv.columns = buffer_cluster_csv.columns.str.strip()

# Seleccionar las variables relevantes
buffer_cluster_csv = buffer_cluster_csv[selected_columns].dropna().copy()

# Codificar las claves categóricas
for col in label_cols:
    buffer_cluster_csv[col] = LabelEncoder().fit_transform(buffer_cluster_csv[col])
    
# Convertir fecha en día de la semana
buffer_cluster_csv['Issue Date'] = pd.to_datetime(buffer_cluster_csv['Issue Date'], errors='coerce')
buffer_cluster_csv['Day of Week'] = buffer_cluster_csv['Issue Date'].dt.dayofweek
buffer_cluster_csv.drop(columns=['Issue Date'], inplace=True)

# Escalar datos
X_scaled = scaler.fit_transform(buffer_cluster_csv)

# K-Means: método del codo y silhouette score
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
# Graficar resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Aplicar K-Means con K=4
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Añadir la etiqueta de cluster al DataFrame original
buffer_cluster_csv['Cluster'] = cluster_labels

# Mostrar cuántas observaciones hay en cada cluster
print("\nNúmero de elementos por cluster:")
print(buffer_cluster_csv['Cluster'].value_counts())

# Ver la media de cada variable por cluster (para interpretación)
print("\nMedia de variables por cluster:")
print(buffer_cluster_csv.groupby('Cluster').mean())

# Gráfico simple (opcional): reducción a 2 dimensiones con PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualización de los clusters en 2D
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
plt.title("Visualización de clusters con PCA (K=4)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show()

# Mostrar número de registros por cluster
print("\nNúmero de observaciones por cluster:")
print(buffer_cluster_csv['Cluster'].value_counts())

# Obtener las medias de cada variable por cluster
medias_cluster = buffer_cluster_csv.groupby('Cluster').mean()

# Redondear para una lectura más clara
medias_cluster = medias_cluster.round(2)

# Mostrar tabla en consola
print("\nMedias por cluster (perfil de cada grupo):")
print(medias_cluster)

#Volumenn de observaciones por cluster
buffer_cluster_csv['Cluster'].value_counts().sort_index().plot(
    kind='bar', color='skyblue', title='Número de infracciones por cluster'
)
plt.xlabel("Cluster")
plt.ylabel("Número de registros")
plt.tight_layout()
plt.show()

#Heatmap de medias por cluster
plt.figure(figsize=(10, 6))
sns.heatmap(medias_cluster, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Media de variables por cluster (K=4)")
plt.tight_layout()
plt.show()

#Metricas del cluster

kmeans.inertia_
silhouette_score(X_scaled, kmeans.labels_)
davies_bouldin_score(X_scaled, kmeans.labels_)
calinski_harabasz_score(X_scaled, kmeans.labels_)

print("Inercia:", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, kmeans.labels_))
print("Davies-Bouldin Index:", davies_bouldin_score(X_scaled, kmeans.labels_))
print("Calinski-Harabasz Index:", calinski_harabasz_score(X_scaled, kmeans.labels_))

# nuevo a revisar

# Conteo por clase
valores_reales = np.bincount(y_test)
valores_predichos = np.bincount(y_pred)

# Asegurar igual longitud
max_len = max(len(valores_reales), len(valores_predichos))
valores_reales = np.pad(valores_reales, (0, max_len - len(valores_reales)))
valores_predichos = np.pad(valores_predichos, (0, max_len - len(valores_predichos)))

# Gráfico comparativo
x = np.arange(max_len)
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, valores_reales, width, label='Reales')
plt.bar(x + width/2, valores_predichos, width, label='Predichos')
plt.xlabel('Clases codificadas (LabelEncoder)')
plt.ylabel('Número de casos')
plt.title('Random Forest - Reales vs. Predichos')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Distribución por cluster
cluster_counts = buffer_cluster_csv['Cluster'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind='bar', color='cornflowerblue')
plt.title("Distribución de observaciones por Cluster (K-Means)")
plt.xlabel("Cluster")
plt.ylabel("Cantidad de observaciones")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

# Mostrar matriz de confusión en porcentaje por clase
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
plt.title("Matriz de Confusión Normalizada (%)")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.tight_layout()
plt.show()

# Convertir valores codificados a etiquetas originales
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Calcular matriz de confusión normalizada
labels_unicos = np.unique(np.concatenate((y_test_labels, y_pred_labels)))
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=labels_unicos)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Crear DataFrame para etiquetar filas y columnas
df_cm = pd.DataFrame(cm_norm, index=labels_unicos, columns=labels_unicos)

# Graficar
plt.figure(figsize=(14, 10))
sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
plt.title("Matriz de Confusión Normalizada (%) - Etiquetas reales")
plt.xlabel("Predicciones")
plt.ylabel("Valores reales")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Lista de variables numericas
variables_cualitativas = [
    'Registration State',
    'Plate Type',
    'Vehicle Make',
    'Vehicle Body Type',
    'Violation Description',
    'Violation County'
]

# Cargar el dataset limpio
df = pd.read_csv("dataset_limpio_final.csv")

# Mostrar cada tabla como imagen con matplotlib
for var in variables_cualitativas:
    if var in df.columns:
        # Frecuencias
        frec_abs = df[var].value_counts().rename("Frecuencia absoluta")
        frec_rel = (df[var].value_counts(normalize=True) * 100).round(2).rename("Frecuencia relativa (%)")
        tabla = pd.concat([frec_abs, frec_rel], axis=1).reset_index()
        tabla.columns = [var, 'Frecuencia absoluta', 'Frecuencia relativa (%)']
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, len(tabla) * 0.5))
        ax.axis('off')
        tabla_plot = ax.table(
            cellText=tabla.values,
            colLabels=tabla.columns,
            cellLoc='center',
            loc='center'
        )
        tabla_plot.auto_set_font_size(False)
        tabla_plot.set_fontsize(9)
        tabla_plot.scale(1, 1.2)
        plt.title(f"Frecuencias de la variable: {var}", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.show()
        

import pandas as pd
import numpy as np

INPUT = "dataset_limpio_final.csv"

# 1) Carga  de columnas duplicadas
df = pd.read_csv(INPUT)
df = df.loc[:, ~df.columns.duplicated()]

# 2) Detección de tipos
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
dt_cols  = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

# Si hay fechas en texto, intenta convertirlas:
for c in df.columns:
    if df[c].dtype == 'object':
        # Heurística: si parece fecha, conviértela
        try_conv = pd.to_datetime(df[c], errors='ignore', infer_datetime_format=True)
        if hasattr(try_conv, 'dt'):
            # si convirtió (hay NaT y fechas), sustituye
            if try_conv.notna().sum() > 0 and (try_conv.dtype == 'datetime64[ns]'):
                df[c] = try_conv
dt_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

# 3) Preparar vistas

# --- NUMÉRICAS: skew y exceso de curtosis (---
num_df = df[num_cols].copy()
# quitar columnas sin variación
num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]

skew_num = num_df.skew(numeric_only=True)
kurt_num = num_df.kurt(numeric_only=True)

tabla_num = pd.DataFrame({
    "variable": skew_num.index,
    "skewness": skew_num.values,
    "kurtosis_excess": kurt_num.reindex(skew_num.index).values,
    "interpretacion": ""
})

def interp_skew(s):
    if pd.isna(s): return ""
    s = float(s)
    if abs(s) < 0.1: return "≈ simétrica"
    if s >= 0.1 and s < 0.5: return "asimetría + leve (cola derecha)"
    if s >= 0.5: return "asimetría + marcada (cola derecha)"
    if s <= -0.1 and s > -0.5: return "asimetría − leve (cola izquierda)"
    return "asimetría − marcada (cola izquierda)"

def interp_kurt(k):
    if pd.isna(k): return ""
    k = float(k)
    if -0.5 <= k <= 0.5: return "curtosis ≈ normal"
    if k > 0.5: return "leptocúrtica (colas pesadas)"
    return "platicúrtica (colas ligeras)"

tabla_num["interp_skew"] = tabla_num["skewness"].apply(interp_skew)
tabla_num["interp_kurt"] = tabla_num["kurtosis_excess"].apply(interp_kurt)

# Orden por |skew|
tabla_num = tabla_num.sort_values("skewness", key=lambda s: s.abs(), ascending=False)

# --- FECHAS: convertir a número (timestamp en días) y tratar como numéricas ---
fecha_results = []
for c in dt_cols:
    s = df[c].dropna()
    if s.empty: 
        continue
    # convertir a días (epoch días)
    vals = (s.view('int64') / 1e9 / 86400.0)  # ns -> s -> días
    skew = pd.Series(vals).skew()
    kurt = pd.Series(vals).kurt()
    fecha_results.append({
        "variable": c,
        "skewness": skew,
        "kurtosis_excess": kurt,
        "interp_skew": interp_skew(skew),
        "interp_kurt": interp_kurt(kurt),
        "nota": "Fecha convertida a escala numérica (días desde epoch)"
    })
tabla_fechas = pd.DataFrame(fecha_results)

# --- CATEGÓRICAS: dos vistas ---
cat_freq_rows = []
cat_codes_rows = []

for c in obj_cols:
    s = df[c].dropna()
    if s.empty:
        continue

    # (A) Vista por FRECUENCIAS de categorías
    freq = s.value_counts()  # conteos por categoría
    # Necesitamos al menos 2 categorías para forma
    if freq.shape[0] > 1:
        skew_f = freq.skew()
        kurt_f = freq.kurt()
        cat_freq_rows.append({
            "variable": c,
            "skewness_freq": skew_f,
            "kurtosis_excess_freq": kurt_f,
            "interp_skew_freq": interp_skew(skew_f),
            "interp_kurt_freq": interp_kurt(kurt_f),
            "advertencia": "Interpretar como forma de la distribución de frecuencias por categoría (no magnitud)."
        })

    # (B) Vista por CÓDIGOS ordinales 
    cat = pd.Categorical(s)
    codes = pd.Series(cat.codes).replace(-1, np.nan).dropna()
    if codes.nunique() > 1:
        skew_c = codes.skew()
        kurt_c = codes.kurt()
        cat_codes_rows.append({
            "variable": c,
            "skewness_codes": skew_c,
            "kurtosis_excess_codes": kurt_c,
            "interp_skew_codes": interp_skew(skew_c),
            "interp_kurt_codes": interp_kurt(kurt_c),
            "advertencia": "Codificación ordinal artificial; valores NO representan magnitud."
        })

tabla_cat_freq  = pd.DataFrame(cat_freq_rows)
tabla_cat_codes = pd.DataFrame(cat_codes_rows)

# 4) Guardar resultados
tabla_num.to_csv("skew_kurtosis_numericas.csv", index=False)
if not tabla_fechas.empty:
    tabla_fechas.to_csv("skew_kurtosis_fechas.csv", index=False)
if not tabla_cat_freq.empty:
    tabla_cat_freq.to_csv("skew_kurtosis_categoricas_frecuencias.csv", index=False)
if not tabla_cat_codes.empty:
    tabla_cat_codes.to_csv("skew_kurtosis_categoricas_codigos.csv", index=False)

# 5) Mostrar por pantalla
print("\nNUMÉRICAS (interpretables):")
print(tabla_num.round(3).to_string(index=False))

if not tabla_fechas.empty:
    print("\nFECHAS (convertidas a días epoch):")
    print(tabla_fechas.round(3).to_string(index=False))

if not tabla_cat_freq.empty:
    print("\nCATEGÓRICAS – Vista por FRECUENCIAS (forma de conteos por categoría):")
    print(tabla_cat_freq.round(3).to_string(index=False))

if not tabla_cat_codes.empty:
    print("\nCATEGÓRICAS – Vista por CÓDIGOS (ordinal artificial; usar con precaución):")
    print(tabla_cat_codes.round(3).to_string(index=False))

import pandas as pd



# --- Cálculo ---
skew = num_df.skew(numeric_only=True)
kurt = num_df.kurt(numeric_only=True)  # exceso de curtosis (Fisher; normal = 0)

# --- Construir tabla ---
tabla = pd.DataFrame({
    "Variable": skew.index,
    "Skewness": skew.values,
    "Kurtosis_Excess": kurt.reindex(skew.index).values
})

# --- Ordenar por magnitud de la asimetría ---
tabla = tabla.sort_values("Skewness", key=lambda s: s.abs(), ascending=False)

# --- Mostrar por pantalla ---
print("\nTabla de asimetría y curtosis (exceso):")
print(tabla.round(3).to_string(index=False))

# --- Guardar en CSV ---
tabla.round(6).to_csv("tabla_skew_kurtosis.csv", index=False)
print("\nGuardado en: tabla_skew_kurtosis.csv")



import pandas as pd
import numpy as np

# === Parámetros ===
INPUT = "dataset_limpio_final.csv"
OUT_TENDENCIA = "tabla_tendencia_central.csv"
OUT_POSICION  = "tabla_posicion.csv"
OUT_DISPERSION = "tabla_dispersion.csv"
OUT_MODA_CATEG = "tabla_moda_categoricas.csv"

# === Utilidades ===
def modos(series, max_items=3):
    """Devuelve hasta max_items modas (como texto concatenado) y el conteo de la primera moda."""
    s = series.dropna()
    if s.empty:
        return pd.NA, 0
    vc = s.value_counts()
    top_vals = vc.index[:max_items]
    top_str = " | ".join(map(str, top_vals))
    return top_str, int(vc.iloc[0])

def to_numeric_epoch_days(dt_series):
    """
    Convierte una serie datetime a días desde epoch (float) para poder calcular
    varianza, std, MAD, etc. No modifica la original.
    """
    s = dt_series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    # astype('int64') -> nanosegundos desde epoch
    ns = s.astype('int64')
    days = ns / 1e9 / 86400.0
    out = pd.Series(days, index=s.index)
    return out

# === Carga ===
df = pd.read_csv(INPUT)
# Elimina columnas duplicadas manteniendo la primera aparición
df = df.loc[:, ~df.columns.duplicated()]

# Tipos
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
obj_cols = df.select_dtypes(include=['object']).columns.tolist()

# Intentar convertir a datetime columnas object que realmente son fechas
dt_cols_detectadas = []
for c in obj_cols:
    try:
        converted = pd.to_datetime(df[c], errors='raise')
        # Si convierte bien y hay al menos algún valor no nulo, sustituimos y registramos
        if converted.notna().sum() > 0:
            df[c] = converted
            dt_cols_detectadas.append(c)
    except Exception:
        pass

dt_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

# === Selección final de columnas por tipo ===
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
obj_cols = df.select_dtypes(include=['object']).columns.tolist()
dt_cols  = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

# Quitar columnas numéricas sin variación
num_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1]

# === TENDENCIA CENTRAL ===
tend_rows = []

# Numéricas
for c in num_cols:
    s = df[c]
    media = s.mean()
    mediana = s.median()
    moda_val, moda_freq = modos(s, max_items=1)
    tend_rows.append({
        "variable": c,
        "tipo": "numérica",
        "media": media,
        "mediana": mediana,
        "moda": moda_val,
        "frecuencia_moda": moda_freq
    })

# Fechas (se puede calcular mediana/percentiles directamente; la media la expresamos como fecha)
for c in dt_cols:
    s = df[c].dropna()
    if s.empty:
        continue
    mediana = s.median()  # pandas permite median en datetime
    # Para la media transformamos a días, sacamos media y devolvemos como fecha aproximada
    epoch_days = to_numeric_epoch_days(s)
    media_days = epoch_days.mean()
    fecha_media = pd.to_datetime(media_days * 86400.0, unit='s')  # vuelta desde días a timestamp
    moda_val, moda_freq = modos(s.dt.date)  # moda por fecha (día)
    tend_rows.append({
        "variable": c,
        "tipo": "fecha",
        "media": fecha_media,
        "mediana": mediana,
        "moda": moda_val,
        "frecuencia_moda": moda_freq
    })

# Categóricas: solo moda (tendencia central)
moda_cat_rows = []
for c in obj_cols:
    moda_val, moda_freq = modos(df[c])
    moda_cat_rows.append({
        "variable": c,
        "tipo": "categórica",
        "moda": moda_val,
        "frecuencia_moda": moda_freq
    })

tabla_tend = pd.DataFrame(tend_rows)
tabla_moda_categoricas = pd.DataFrame(moda_cat_rows)

# === POSICIÓN ===
pos_rows = []

# Numéricas
for c in num_cols:
    s = df[c].dropna()
    pos_rows.append({
        "variable": c,
        "tipo": "numérica",
        "min": s.min(),
        "percentil_10": s.quantile(0.10),
        "cuartil_1 (P25)": s.quantile(0.25),
        "mediana (P50)": s.quantile(0.50),
        "cuartil_3 (P75)": s.quantile(0.75),
        "percentil_90": s.quantile(0.90),
        "max": s.max()
    })

# Fechas (pandas soporta quantile en datetime)
for c in dt_cols:
    s = df[c].dropna()
    if s.empty:
        continue
    pos_rows.append({
        "variable": c,
        "tipo": "fecha",
        "min": s.min(),
        "percentil_10": s.quantile(0.10),
        "cuartil_1 (P25)": s.quantile(0.25),
        "mediana (P50)": s.quantile(0.50),
        "cuartil_3 (P75)": s.quantile(0.75),
        "percentil_90": s.quantile(0.90),
        "max": s.max()
    })

tabla_pos = pd.DataFrame(pos_rows)

# === DISPERSIÓN ===
disp_rows = []

# Numéricas
for c in num_cols:
    s = df[c].dropna()
    if s.empty:
        continue
    var = s.var(ddof=1)               # varianza muestral
    std = s.std(ddof=1)               # desviación estándar muestral
    iqr = s.quantile(0.75) - s.quantile(0.25)
    rango = s.max() - s.min()
    mad = (s - s.median()).abs().median()  # MAD
    cv = (std / s.mean()) if s.mean() != 0 else np.nan  # coeficiente de variación
    disp_rows.append({
        "variable": c,
        "tipo": "numérica",
        "varianza (ddof=1)": var,
        "desviación_estándar (ddof=1)": std,
        "IQR (P75-P25)": iqr,
        "rango (max-min)": rango,
        "MAD": mad,
        "coef_variación (std/mean)": cv
    })

# Fechas: convertimos a días epoch para estimar dispersión
for c in dt_cols:
    sdt = df[c].dropna()
    if sdt.empty:
        continue
    s = to_numeric_epoch_days(sdt)  # días como float
    var = s.var(ddof=1)
    std = s.std(ddof=1)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    rango = s.max() - s.min()
    mad = (s - s.median()).abs().median()
    cv = (std / s.mean()) if s.mean() != 0 else np.nan
    disp_rows.append({
        "variable": c,
        "tipo": "fecha (días epoch)",
        "varianza (ddof=1)": var,
        "desviación_estándar (ddof=1)": std,
        "IQR (P75-P25)": iqr,
        "rango (max-min)": rango,
        "MAD": mad,
        "coef_variación (std/mean)": cv
    })

tabla_disp = pd.DataFrame(disp_rows)

# === Guardado ===
tabla_tend.to_csv(OUT_TENDENCIA, index=False)
tabla_pos.to_csv(OUT_POSICION, index=False)
tabla_disp.to_csv(OUT_DISPERSION, index=False)
tabla_moda_categoricas.to_csv(OUT_MODA_CATEG, index=False)

print(f"Guardadas: \n- {OUT_TENDENCIA}\n- {OUT_POSICION}\n- {OUT_DISPERSION}\n- {OUT_MODA_CATEG}")


import pandas as pd

# Lista de archivos a mostrar
archivos = [
    "tabla_tendencia_central.csv",
    "tabla_posicion.csv",
    "tabla_dispersion.csv",
    "tabla_moda_categoricas.csv"
]

# Mostrar cada tabla en consola
for archivo in archivos:
    print(f"\nContenido de {archivo}:")
    try:
        df = pd.read_csv(archivo)
        print(df.to_string(index=False))  # Muestra todo sin índice
    except Exception as e:
        print(f"Error al leer {archivo}: {e}")
