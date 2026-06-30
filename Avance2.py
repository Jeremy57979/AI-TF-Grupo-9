import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cargamos la data
df_residuos = pd.read_csv('1. Generación Anual de residuos municipal_Distrital_2014_2024.csv', delimiter=';', encoding='latin-1')

# exploramos nuestras columnas
print("Variables del Entorno:")
print(df_residuos.columns.tolist())

# curacion y filtrado
df_limpio = df_residuos[(df_residuos['POB_TOTAL'] > 0) & (df_residuos['QRESIDUOS_MUN'] > 0)].copy()

print(f"Registros originales: {df_residuos.shape[0]}")
print(f"Registros limpios para ML: {df_limpio.shape[0]}")

# Tasa de Urbanización (Qué tan urbana es la zona)
df_limpio['TASA_URBANIZACION'] = df_limpio['POB_URBANA'] / df_limpio['POB_TOTAL']

# Generación Per Cápita (kg por persona al año)
df_limpio['KG_PER_CAPITA'] = (df_limpio['QRESIDUOS_MUN'] * 1000) / df_limpio['POB_TOTAL']

# Vista de las variables preparadas
columnas_ml = ['PERIODO', 'REG_NAT', 'DISTRITO', 'TASA_URBANIZACION', 'KG_PER_CAPITA']
print(df_limpio[columnas_ml].head())

# Agrupación temporal
tendencia_anual = df_limpio.groupby('PERIODO')['QRESIDUOS_MUN'].sum().reset_index()
tendencia_anual.rename(columns={'QRESIDUOS_MUN': 'TOTAL_TONELADAS_NACIONAL'}, inplace=True)

print("Matriz temporal lista para el algoritmo predictivo:")
print(tendencia_anual)

# Gráfico
plt.figure(figsize=(12, 6))

plt.plot(tendencia_anual['PERIODO'], 
         tendencia_anual['TOTAL_TONELADAS_NACIONAL'], 
         marker='o', 
         linestyle='-', 
         color='#005088', 
         linewidth=2, 
         markersize=8)

plt.title('Evolución de la Generación de Residuos Sólidos a Nivel Nacional (2014-2024)', fontsize=14, pad=15)
plt.xlabel('Año de Registro (PERIODO)', fontsize=12)
plt.ylabel('Toneladas Totales Generadas', fontsize=12)

plt.ticklabel_format(style='plain', axis='y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(tendencia_anual['PERIODO'])

plt.tight_layout()
plt.show()
# Análisis por Región Natural
print("\nDistribución histórica por Región Natural (2014-2024):")
distribucion_region = df_limpio.groupby('REG_NAT')['QRESIDUOS_MUN'].sum().reset_index()

# Ordenamos de mayor a menor
distribucion_region = distribucion_region.sort_values(by='QRESIDUOS_MUN', ascending=False)
print(distribucion_region)

#almacenamos el df limpio
print("\nGuardando la base de datos ")

df_limpio.to_csv('dataset_limpio.csv', index=False, encoding='latin-1', sep=';')

#grafico
# --- GRÁFICO: Distribución por Región Natural ---
plt.figure(figsize=(10, 6))

# Creamos el gráfico de barras con colores diferenciados para cada región
plt.bar(distribucion_region['REG_NAT'],
        distribucion_region['QRESIDUOS_MUN'],
        color=['#e67e22', '#27ae60', '#2980b9'],
        edgecolor='black')

plt.title('Distribución Histórica de Residuos por Región Natural (2014-2024)', fontsize=14, pad=15)
plt.xlabel('Región Natural', fontsize=12)
plt.ylabel('Toneladas Totales Generadas', fontsize=12)

# Formateo para evitar notación científica
plt.ticklabel_format(style='plain', axis='y')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Guardamos también la matriz temporal para las Series de Tiempo
tendencia_anual.to_csv('tendencia_anual_nacional.csv', index=False, encoding='latin-1', sep=';')

print("Base de datos almacenada")



# CLUSTERING CON K-MEANS


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Variables logarítmicas para reducir el efecto de valores extremos
df_limpio['LOG_POB_TOTAL'] = np.log1p(df_limpio['POB_TOTAL'])
df_limpio['LOG_KG_PER_CAPITA'] = np.log1p(df_limpio['KG_PER_CAPITA'])

# Variables que usará el modelo
variables_cluster = df_limpio[
    [
        'LOG_POB_TOTAL',
        'TASA_URBANIZACION',
        'LOG_KG_PER_CAPITA'
    ]
].copy()

# Normalización de variables
scaler = StandardScaler()
X_cluster = scaler.fit_transform(variables_cluster)

# Aplicación de K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_limpio['CLUSTER'] = kmeans.fit_predict(X_cluster)


from sklearn.metrics import silhouette_score

# Evaluación del modelo K-Means
silhouette = silhouette_score(X_cluster, df_limpio['CLUSTER'])

print("\nEvaluación del modelo K-Means:")
print(f"Silhouette Score: {silhouette:.4f}")

if silhouette >= 0.70:
    print("Interpretación: Excelente separación entre clusters.")
elif silhouette >= 0.50:
    print("Interpretación: Buena separación entre clusters.")
elif silhouette >= 0.25:
    print("Interpretación: Separación aceptable entre clusters.")
else:
    print("Interpretación: Separación débil entre clusters.")

# Resumen de clusters
resumen_clusters = df_limpio.groupby('CLUSTER')[
    [
        'POB_TOTAL',
        'TASA_URBANIZACION',
        'KG_PER_CAPITA',
        'QRESIDUOS_MUN'
    ]
].mean()

print("\nResumen promedio por cluster:")
print(resumen_clusters)

print("\nCantidad de registros por cluster:")
print(df_limpio['CLUSTER'].value_counts())

# Etiquetas interpretativas
df_limpio['TIPO_CLUSTER'] = df_limpio['CLUSTER'].map({
    0: 'Intermedio',
    1: 'Rural / baja generación',
    2: 'Urbano / alta generación'
})

# Gráfico del clustering
plt.figure(figsize=(10, 6))

plt.scatter(
    df_limpio['TASA_URBANIZACION'],
    df_limpio['KG_PER_CAPITA'],
    c=df_limpio['CLUSTER'],
    cmap='viridis',
    alpha=0.6
)

plt.title('Clustering de Distritos según Urbanización y Generación Per Cápita', fontsize=14, pad=15)
plt.xlabel('Tasa de Urbanización', fontsize=12)
plt.ylabel('Kg de Residuos Per Cápita', fontsize=12)
plt.colorbar(label='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('clustering_distritos.png', dpi=300)
print("Gráfico de clustering guardado como 'clustering_distritos.png'")

plt.show()

# Guardamos el dataset con los clusters
df_limpio.to_csv(
    'dataset_limpio_con_clusters.csv',
    index=False,
    encoding='latin-1',
    sep=';'
)

print("Dataset con clusters almacenado correctamente.")
