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

