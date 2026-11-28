# Prioridad de adeudos

Prototipo académico para explorar una lista de adeudos sintéticos. Ordena por la probabilidad estimada de recuperación y compara ese orden con monto pendiente y días de atraso. Es una herramienta de demostración, no una recomendación para decisiones fiscales reales.

## Puesta en marcha en Windows

Requiere Python 3.12. Desde la raíz del proyecto en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m backend.train
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Abrir http://127.0.0.1:8000. Para detener el servidor, usar Ctrl+C. El modelo se genera localmente en `backend/artifacts/`, carpeta excluida de Git. Si falta el artefacto, ejecutar el entrenamiento antes de analizar.

Para comprobar la API y el modelo: `.\.venv\Scripts\python.exe -m pytest backend/tests`.

## Datos y modelo

La carpeta de Drive suministrada contenía dos cuadernos: generación de datos y entrenamiento. No contenía el `.pkl` exportado por el cuaderno. Los cuadernos locales conservan su código, sin salidas ni metadatos de ejecución. El comando de entrenamiento reconstruye un modelo nuevo a partir del generador sintético con semilla 42, filtro `situacion_adeudo=1`, división estratificada 80/20, escala de variables numéricas, codificación de respuesta y Random Forest de 150 árboles y profundidad máxima 8. No es el archivo entrenado original. La muestra de la interfaz usa otra semilla y no forma parte del conjunto de entrenamiento.

El objetivo `recuperado` se construye con una regla y ruido sintéticos; por tanto, las métricas y probabilidades describen esa simulación y no demuestran calibración o eficacia en datos reales. El orden por modelo es probabilidad descendente de recuperación estimada, no una medición de urgencia fiscal ni de impacto esperado. Los empates se resuelven por identificador. No se muestran explicaciones individuales causales. Casos con situación 0 requieren revisión y se excluyen de los tres ordenamientos.

El CSV debe tener encabezados `id_deudor,monto_pendiente,dias_vencimiento,incumplimientos_anteriores,pagos_parciales,intentos_contacto,respuesta_contacto,situacion_adeudo`. Los montos son números no negativos en MXN; días y conteos son enteros no negativos. Respuesta: 0 sin respuesta, 1 sin compromiso, 2 con compromiso. Situación: 1 disponible para cobro, 0 requiere revisión. Identificadores únicos. No incluyas datos reales ni sensibles.

## Decisiones y límites

FastAPI mantiene la inferencia en el mismo entorno Python que scikit-learn; la interfaz estática evita un proceso de compilación separado. Los límites y parámetros del generador no representan una población real. El sistema no persiste cargas de usuarios. Para integrar el artefacto original harían falta su archivo, versiones exactas de entrenamiento y prueba de equivalencia de predicciones; no debe sustituirse silenciosamente por un archivo externo no verificado.

La identidad visual retoma la presentación del proyecto: fondo violeta oscuro, acentos rosa y turquesa, y tipografía de alto contraste. Es una simulación académica independiente inspirada en la cobranza fiscal y no representa un servicio oficial.

Para la presentación, pulsa **Analizar muestra**, compara el orden original de cada caso con sus posiciones por modelo, saldo y días, y abre su detalle. Los números originales conservan la posición del archivo incluso cuando hay casos excluidos. Una probabilidad más alta indica mayor recuperación estimada en la simulación; no equivale a mayor urgencia ni es una explicación causal. La muestra integrada es independiente y procede del mismo generador. El usuario proporcionó después los CSV originales: las 5.000 filas de entrenamiento coinciden por valores con los datos reproducidos, y el CSV original de demostración contiene 30 filas de ese mismo conjunto (22 usadas para entrenar, 5 para la prueba y 3 excluidas por su situación). Puede cargarse para demostración, pero no debe presentarse como evaluación independiente.
