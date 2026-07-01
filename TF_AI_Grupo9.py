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

# ==========================================
# MODELO PREDICTIVO - REGRESIÓN LINEAL
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Variables del modelo
X = tendencia_anual[['PERIODO']]
y = tendencia_anual['TOTAL_TONELADAS_NACIONAL']

# Separación cronológica de datos
# Entrenamiento: 2014 - 2021
# Prueba: 2022 - 2024
X_train = X.iloc[:8]
y_train = y.iloc[:8]

X_test = X.iloc[8:]
y_test = y.iloc[8:]

# Crear y entrenar el modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# Predicción sobre datos de prueba
y_pred = modelo_regresion.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo de Regresión Lineal:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Comparación real vs predicho
comparacion = pd.DataFrame({
    'PERIODO': X_test['PERIODO'].values,
    'VALOR_REAL': y_test.values,
    'VALOR_PREDICHO': y_pred
})

print("\nComparación de valores reales vs predichos:")
print(comparacion)

# Entrenar modelo final con todos los datos disponibles
modelo_final = LinearRegression()
modelo_final.fit(X, y)

# Predicción futura
periodos_futuros = pd.DataFrame({
    'PERIODO': [2025, 2026, 2027]
})

predicciones_futuras = modelo_final.predict(periodos_futuros)

resultado_prediccion = pd.DataFrame({
    'PERIODO': periodos_futuros['PERIODO'],
    'PREDICCION_TONELADAS': predicciones_futuras
})

print("\nPredicción de generación de residuos:")
print(resultado_prediccion)

# Gráfico de datos reales, prueba y predicción futura
plt.figure(figsize=(12, 6))

plt.plot(
    tendencia_anual['PERIODO'],
    tendencia_anual['TOTAL_TONELADAS_NACIONAL'],
    marker='o',
    linestyle='-',
    label='Datos reales'
)

plt.plot(
    X_test['PERIODO'],
    y_pred,
    marker='o',
    linestyle='--',
    label='Predicción en prueba'
)

plt.plot(
    resultado_prediccion['PERIODO'],
    resultado_prediccion['PREDICCION_TONELADAS'],
    marker='o',
    linestyle='--',
    label='Predicción futura'
)

plt.title('Predicción de la Generación de Residuos Sólidos mediante Regresión Lineal', fontsize=14, pad=15)
plt.xlabel('Periodo', fontsize=12)
plt.ylabel('Toneladas Totales Generadas', fontsize=12)
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(list(tendencia_anual['PERIODO']) + [2025, 2026, 2027])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('prediccion_regresion_lineal.png', dpi=300)
plt.show()

print("Gráfico guardado como 'prediccion_regresion_lineal.png'")



# ==========================================
# MODELO PREDICTIVO 2 - RANDOM FOREST REGRESSOR
# ==========================================
from sklearn.ensemble import RandomForestRegressor

# 1. Crear y entrenar el modelo de Random Forest (usando los mismos datos divididos)
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# 2. Predicción sobre datos de prueba
y_pred_rf = modelo_rf.predict(X_test)

# 3. Evaluación del modelo
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n" + "="*50)
print(" EVALUACIÓN DEL MODELO: RANDOM FOREST REGRESSOR")
print("="*50)
print(f"MAE:  {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R²:   {r2_rf:.4f}")
print("-> Diagnóstico técnico: El R² negativo demuestra empíricamente que los")
print("   árboles de decisión no pueden extrapolar proyecciones a futuro cuando")
print("   la tendencia histórica es estrictamente ascendente.\n")

# 4. Comparación real vs predicho (RF)
comparacion_rf = pd.DataFrame({
    'PERIODO': X_test['PERIODO'].values,
    'VALOR_REAL': y_test.values,
    'VALOR_PREDICHO_RF': y_pred_rf
})
print("Comparación de valores en fase de prueba (Random Forest):")
print(comparacion_rf)

# 5. Entrenar modelo final con todos los datos disponibles
modelo_final_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_final_rf.fit(X, y)

# 6. Predicción futura
predicciones_futuras_rf = modelo_final_rf.predict(periodos_futuros)

resultado_prediccion_rf = pd.DataFrame({
    'PERIODO': periodos_futuros['PERIODO'],
    'PREDICCION_TONELADAS_RF': predicciones_futuras_rf
})

print("\nPredicción de generación a futuro (Random Forest):")
print(resultado_prediccion_rf)

# GRÁFICO RANDOM FOREST
plt.figure(figsize=(12, 6))

# Datos Históricos
plt.plot(
    tendencia_anual['PERIODO'],
    tendencia_anual['TOTAL_TONELADAS_NACIONAL'],
    marker='o',
    linestyle='-',
    color='black',
    label='Datos reales (2014-2024)'
)

# Prueba RF
plt.plot(
    X_test['PERIODO'],
    y_pred_rf,
    marker='s',
    linestyle='--',
    color='#e67e22',
    label='Prueba RF (2022-2024)'
)

# Predicción Futura RF 
plt.plot(
    resultado_prediccion_rf['PERIODO'],
    resultado_prediccion_rf['PREDICCION_TONELADAS_RF'],
    marker='X',
    linestyle='--',
    color='#c0392b',
    label='Predicción Futura RF (2025-2027)'
)

plt.title('Falla de Extrapolación Predictiva mediante Random Forest', fontsize=14, pad=15)
plt.xlabel('Periodo', fontsize=12)
plt.ylabel('Toneladas Totales Generadas', fontsize=12)
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(list(tendencia_anual['PERIODO']) + [2025, 2026, 2027])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('prediccion_random_forest.png', dpi=300)
plt.show()

print("\nGráfico de Random Forest guardado exitosamente como 'prediccion_random_forest.png'")

