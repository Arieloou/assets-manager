Genera un plan de implementación para un proyecto de predicción de fallos en equipos electrónicos a través de un modelo de Inteligencia Artificial.
### Guidelines del modelo de IA:
- El modelo de IA elegido es Random Forest
- El dataset elegido tiene la siguiente estructura:

| Columna                        | Tipo                  | Rol Potencial                 |
| ------------------------------ | --------------------- | ----------------------------- |
| `ID_Equipo`                    | string (8,000 únicos) | Identificador — **descartar** |
| `Vida_Util_Consumida`          | float64               | Feature numérica              |
| `Tasa_Incidencias_Tecnicas`    | int64                 | Feature numérica              |
| `Tiempo_Inactividad_Acumulado` | float64               | Feature numérica              |
| `Costo_Mto_Reactivo_Acumulado` | float64               | Feature numérica              |
| `Ubicacion_Activo`             | string (4 categorías) | Feature categórica            |
| `Estado_Integridad_Hardware`   | string (4 categorías) | **Candidato a target**        |
|                                |                       |                               |
- Las variables de entrada que manejará son las siguientes:
	1. `Vida_Util_Consumida`
	2. `Tasa_Incidencias_Tecnicas`
	3. `Tiempo_Inactividad_Acumulado`
	4. `Costo_Mto_Reactivo_Acumulado`
	5. `Campus_Activo (ubicacion)`
	6. `Estado_Integridad_Hardware`
	7. `Tipo_Equipo`
- Las variables target del modelo son: `Estado_Integridad_Hardware` y `Nivel_Riesgo_Operativo`
- Es importante tomar en consideración las siguientes anotaciones para el modelo:
	1. Validar con cross-validation riguroso (k-fold, k≥5)
	2. Analizar la `feature_importance` del modelo entrenado
	3. Probar el modelo excluyendo `Vida_Util_Consumida` para verificar robustez
- **Data Leakage potencial:** Las correlaciones extremadamente altas (0.969) entre `Vida_Util_Consumida` y el target `Estado_Integridad_Hardware`, sugieren que esta variable podría haber sido usada directamente para generar las etiquetas durante la creación del dataset sintético. Esto puede resultar en métricas artificialmente altas.
### Guidelines de funcionalidades:
- El sistema contará con:
	- Implementar un sistema de Login y LogOut
	- Construir un dashboard de análisis de activos actualmente registrados
		- Tiempo de Inactividad Promedio
		- Costo de Mantenimiento Promedio
		- Diagrama de dona para representar la ubicación de los activos clasificados por sedes:
			- UDLAPARK
			- GRANADOS
			- COLON
		Este podrá ser filtrado en base a los siguientes modificadores:
		1. VIDA ÚTIL:
				vida_util: (mean_pct, std_pct, clip_low, clip_high)
				VIDA_UTIL_PARAMS = {
					"ÓPTIMO": (15,  5,  5,  25),
					"OPERATIVO":     (40,  8, 20,  55),
					"PREVENTIVO":   (65,  8, 50,  80),
					"CRÍTICO":   (88,  6, 75, 100),
				}
		2. TASA DE INCIDENCIAS:
			tasa_incidencias: Poisson lambda per state
				INCIDENCIAS_LAMBDA = {
				    "BAJA": 0,
				    "MEDIA":     1,
				    "ALTA":   3,
				    "CRÍTICA":   5,
				}
		3. COSTO DE REPARACIÓN:
			costo_mto: (low_usd, high_usd) uniform range per state
				COSTO_PARAMS = {
				    "BAJO": (0,   30),
				    "MEDIO":     (20,  80),
				    "REGULAR":   (70,  150),
				    "ALTO":   (150, 400),
				}
		4. UBICACIÓN: Según la sede en la que se encuentren los activos.
		5. ESTADO INTEGRIDAD DE HARDWARE: ["Excelente", "Bueno", "Regular", "Crítico"]
		6. TIPO DE EQUIPO: Según la clase de equipo que conste para cada registro
	- Implementar un sistema para importar y exportar archivos en formato .csv (data lista para ser cargada y continuar alimentando el histórico del modelo Random Forest)
		- Cada nuevo registro debe guardarse en el histórico con una marca de tiempo específica para su correcta identificación posteriormente.
	- Implementar un sistema de predicción del estado de integridad de un equipo en base a las features solicitadas por el modelo.
	- **Monitoreo de Data Drift:** Implementar un sistema que analice las distribuciones de las variables de entrada a lo largo del tiempo. Si la distribución de `Vida_Util_Consumida` o los parámetros de `Costo_Mto_Reactivo_Acumulado` en los nuevos archivos CSV difieren significativamente de los datos de entrenamiento originales, el sistema debe alertar sobre una posible degradación en la fiabilidad del modelo.
	- **Matriz de Confusión Dinámica:** Añadir al dashboard una visualización del rendimiento histórico del modelo, permitiendo al usuario auditar si los falsos positivos (predecir un estado crítico cuando es excelente) están aumentando.
	- **Sistema de Alertas Tempranas:** Un módulo que actúe sobre las predicciones de `Estado_Integridad_Hardware` y `Nivel_Riesgo_Operativo`. Si la inferencia arroja un resultado "Crítico" para un equipo de alto valor, el sistema debe registrar una alerta prioritaria en la base de datos.
	- **Análisis de Importancia de Variables en Tiempo Real:** Dado que las guidelines del planteamiento del modelo de IA exigen analizar la `feature_importance`, el dashboard de Streamlit debe exponer una gráfica interactiva que muestre cómo el modelo Random Forest pondera cada variable en la versión activa, reflejando cualquier cambio tras un reentrenamiento.
### Guidelines creación de la aplicación:
- Utiliza Context7 para el framework de Streamlit, sobre el cual se construirá la aplicación.
- Toda la data generada en la aplicación, deberá ser almacenada en una base de datos PostgreSQL a través de la documentación oficial de Streamlit.
- Utiliza una arquitectura basada en características para modularizar la aplicación en features claramente defnidas.