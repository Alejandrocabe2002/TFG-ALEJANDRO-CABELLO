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
