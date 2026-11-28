# De los adeudos a las prioridades

Aplicación web académica que explora cómo priorizar una cartera de adeudos para apoyar la cobranza fiscal. A partir de datos sintéticos, un modelo Random Forest estima la probabilidad de recuperación y propone un orden de atención. La interfaz permite comparar ese orden con el archivo original y con ordenamientos por monto o días de atraso.

> **Prototipo académico:** utiliza datos sintéticos y no está afiliado ni avalado por el SAT. Sus resultados no deben utilizarse para tomar decisiones reales de cobranza.

## ¿Qué permite hacer?

- Analizar una muestra sintética incluida en la aplicación o cargar un archivo CSV con el formato esperado.
- Comparar el orden original de los casos con el orden propuesto por el modelo, por saldo o por días vencidos.
- Consultar el detalle de cada caso y una explicación de sensibilidad calculada con el modelo local.
- Comparar los primeros 50 casos de cada estrategia en un conjunto sintético separado para prueba.

## Requisitos

- Python 3.12 y Git instalados.
- PowerShell en Windows y conexión a internet para descargar las dependencias.
- Un navegador moderno. La interfaz y la API se ejecutan juntas; no es necesario instalar Node.js.

## Instalación y primer inicio

1. Clona el repositorio y entra a la carpeta:

   ```powershell
   git clone https://github.com/EreAguilaReal/Proyecto_IA.git
   cd Proyecto_IA
   ```

2. Crea un entorno virtual e instala las dependencias:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

3. Genera el modelo local. Este paso entrena el pipeline con los datos sintéticos y prepara la comparación de prueba:

   ```powershell
   .\.venv\Scripts\python.exe -m backend.train
   ```

4. Inicia la aplicación:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

5. Abre **[http://127.0.0.1:8000](http://127.0.0.1:8000)** en el navegador. Deja la terminal abierta mientras uses la aplicación; para detener el servidor, vuelve a ella y presiona `Ctrl+C`.

Para volver a iniciar el proyecto, abre PowerShell en la carpeta `Proyecto_IA` y ejecuta únicamente el comando del paso 4. Repite el entrenamiento si aún no existe el modelo local o si deseas regenerarlo. El archivo generado se guarda en `backend/artifacts/` y no se incluye en Git.

## Cómo usar la aplicación

1. Selecciona **Analizar muestra** para utilizar 30 casos sintéticos de ejemplo, o carga un CSV sintético.
2. Compara la lista del archivo con la lista de la derecha. Puedes cambiar el criterio entre modelo, saldo y días vencidos.
3. Busca un identificador completo o selecciona **Detalle** para revisar los datos y la sensibilidad de la estimación.
4. En la sección de evaluación, compara cuántos casos etiquetados como recuperados aparecen entre los primeros 50 de cada ordenamiento.

Las posiciones del archivo conservan huecos cuando se excluyen casos que requieren revisión. El movimiento de posición compara únicamente los casos disponibles para cobro.

## Formato del CSV

El archivo debe incluir los siguientes encabezados y al menos un caso. Se aceptan separadores de coma o punto y coma; usa punto decimal para los importes.

| Campo | Descripción |
| --- | --- |
| `id_deudor` | Identificador único del caso. |
| `monto_pendiente` | Saldo pendiente no negativo, en MXN. |
| `dias_vencimiento` | Número entero de días vencidos. |
| `incumplimientos_anteriores` | Número entero de incumplimientos previos. |
| `pagos_parciales` | Importe no negativo abonado previamente, en MXN. |
| `intentos_contacto` | Número entero de intentos de contacto. |
| `respuesta_contacto` | `0`: sin respuesta; `1`: sin compromiso; `2`: con compromiso. Con cero intentos debe ser `0`. |
| `situacion_adeudo` | `1`: disponible para cobro; `0`: requiere revisión y queda fuera del ordenamiento. |

Ejemplo con casos ficticios:

```csv
id_deudor,monto_pendiente,dias_vencimiento,incumplimientos_anteriores,pagos_parciales,intentos_contacto,respuesta_contacto,situacion_adeudo
DEMO-001,5000,90,1,1000,2,2,1
DEMO-002,12000,160,2,0,0,0,1
```

No cargues información personal ni datos fiscales reales para esta demostración.

## Modelo y evaluación

El pipeline utiliza un Random Forest de 150 árboles y profundidad máxima 8. Antes de la predicción calcula la proporción abonada, escala las variables numéricas y codifica la respuesta de contacto. El entrenamiento parte de 5,000 registros sintéticos; filtra los casos disponibles para cobro y separa los datos en entrenamiento y prueba con una división estratificada de 80/20.

La prioridad mostrada corresponde a la **probabilidad estimada de recuperación sintética**, en orden descendente. No mide por sí sola la urgencia fiscal ni garantiza que un caso vaya a pagar. Los empates exactos se resuelven por identificador. El detalle compara la predicción del caso con escenarios en los que se sustituyen grupos de variables por referencias del entrenamiento: son cambios aislados, no causas ni contribuciones que puedan sumarse.

La evaluación retrospectiva utiliza los mismos 897 casos sintéticos de prueba para cada estrategia y cuenta cuántos de los primeros 50 tienen la etiqueta `recuperado=1`. Esta comparación describe el experimento sintético y no mide resultados de gestiones de cobranza reales.

## Pruebas

Con las dependencias instaladas, ejecuta desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Las pruebas comprueban el entrenamiento, la API, las validaciones, la comparación y la explicación del modelo.

## Estructura del proyecto

```text
backend/          API, generación sintética, entrenamiento y pruebas
frontend/         Interfaz web y recursos visuales
datos.ipynb       Generación de datos sintéticos
modelo.ipynb      Entrenamiento y comparación de modelos
```

El logotipo del SAT se incluye como referencia de inspiración a partir de un [documento oficial](https://www.sat.gob.mx/minisitio/Transparencia/documentos/normatividad_transparencia/Lineamientos_Acceso_Info_Transparencia_SAT.pdf). El proyecto es independiente y no cuenta con afiliación ni aval de esa institución.
