# TP2 — Detección y Clasificación de Pastillas en Cinta Transportadora

**Materia:** Procesamiento Digital de Imágenes I  
**Carrera:** Tecnicatura Universitaria en Inteligencia Artificial (TUIA) — UNR  

---

## Descripción del problema

Se dispone de una imagen de una cinta transportadora industrial con pastillas de distintos tipos mezcladas sobre un fondo oscuro. El objetivo es, sin usar deep learning ni OCR externo, **detectar**, **segmentar** y **clasificar automáticamente** cada pastilla por tipo, reportando la cantidad de cada uno.

Los cinco tipos de pastillas presentes son:

| Código | Descripción |
|--------|-------------|
| `BR` | Blanca Redonda |
| `BC` | Blanca Cuadrada |
| `AP` | Amarilla (cápsula) |
| `RR` | Rosada Redonda |
| `AzC` | Azul-Celeste (cápsula mitad azul, mitad blanca) |

---

## Pipeline de procesamiento

El pipeline se divide en cuatro etapas:

### A — Detección de ROI (región de interés)

La imagen contiene el marco metálico de la cinta, que no es relevante para el análisis. La ROI se detecta **automáticamente** sin valores hardcodeados:

- Se calcula el perfil de intensidad promedio por fila (`np.mean` sobre el eje horizontal).
- Se suaviza con un promedio móvil de ventana 20 para eliminar variaciones pequeñas.
- Se calcula la derivada discreta del perfil suavizado.
- El borde superior es la caída más brusca en la primera mitad (metal brillante → cinta oscura).
- El borde inferior es la subida más brusca en la segunda mitad (cinta oscura → metal brillante).

### B — Segmentación de pastillas

Con la ROI recortada, se separan las pastillas del fondo oscuro:

- Conversión a HSV y extracción del canal **V** (brillo). Las pastillas son objetos brillantes sobre fondo oscuro, por lo que V es el canal más discriminativo.
- Umbral dinámico basado en **percentil 88** del canal V. Esto equivale a "el 12% más brillante de la cinta son pastillas", adaptándose a variaciones de iluminación entre imágenes.
- Cierre morfológico con kernel elíptico 5×5 para rellenar huecos internos causados por reflejos.
- Filtrado de contornos por:
  - Área mínima: 0.14% del área de la ROI (derivado del tamaño relativo de una pastilla).
  - Relación de aspecto: se descartan contornos con `w/h > 10` (bordes horizontales de la ROI).

### C — Clasificación por color y forma

#### Clasificación por color (LAB + HSV)

Cada pastilla se clasifica individualmente usando sus propios promedios en espacio **CIELAB** y **HSV**:

- **Canal A** (eje verde–rojo): `A > 140` → rosada (`RR`)
- **Canal B** (eje azul–amarillo): `B > 138` → amarilla (`AP`); `B < 120` → azul (`AzC`)
- Si A y B son neutros: se valida con HSV que `V > 120` y `S < 60` → blanca

Se eligió CIELAB porque el canal L (luminosidad) está desacoplado de los canales cromáticos, lo que hace la clasificación robusta frente a variaciones de iluminación. Se descartó K-means por su aleatoriedad en la inicialización de centroides.

#### Clasificación por forma (solo blancas)

Las pastillas blancas se subdividen en redondas y cuadradas usando la **circularidad geométrica**:

```
circularidad = 4 * π * área / perímetro²
```

- Círculo perfecto → 1.0
- Cuadrado ideal → π/4 ≈ 0.785
- Umbral aplicado: `circularidad > 0.88` → `BR`, sino → `BC`

Se descartó K-means sobre circularidades por la misma razón: resultados no determinísticos. El umbral fijo es justificable geométricamente.

### D — Resultado final

Se genera una imagen con contornos verdes y etiquetas `TIPO + ID` sobre cada pastilla, y se reporta el conteo por tipo por consola.

---

## Problemáticas encontradas y soluciones

### Umbral de segmentación dependiente de la imagen

**Problema:** un umbral fijo en el canal V (originalmente `110`) fallaba cuando la imagen tenía condiciones de iluminación distintas.  
**Solución:** reemplazar el valor fijo por el **percentil 88** del canal V calculado sobre la ROI. El umbral se adapta automáticamente a cada imagen.

### Clasificación por color no determinística con K-means

**Problema:** K-means con `K=4` sobre colores promedio LAB producía resultados distintos en cada ejecución por la inicialización aleatoria de centroides. Además, requería una función extra para interpretar los centros resultantes.  
**Solución:** clasificación determinística pastilla por pastilla usando umbrales directos sobre los canales A y B de CIELAB, validando con HSV para las blancas. Mismo resultado en cada ejecución, sin aleatoriedad.

### Separación blancas redondas vs cuadradas con K-means

**Problema:** K-means sobre circularidades también era no determinístico y podía invertir los grupos entre ejecuciones.  
**Solución:** umbral fijo de circularidad `0.88`, justificado geométricamente: los valores ideales son `1.0` para círculos y `0.785` para cuadrados, con `0.88` como separador natural entre ambos grupos.

### Pastillas azul-celeste con canal B cercano al neutro

**Problema:** las cápsulas AzC son mitad blancas y mitad azules. El promedio del canal B no cae tan por debajo de 128 como en un objeto completamente azul.  
**Solución:** umbral asimétrico: `B < 120` (en lugar de `< 128`) para capturar el desplazamiento parcial del promedio hacia el azul.

---

## Resultados

![Imagen original](pills.png)

![Segmentación — paso B](resultado_B.png)

![Clasificación final — paso D](resultado_D.png)

### Conteo obtenido

| Tipo | Cantidad |
|------|----------|
| BR — Blanca Redonda | 13 |
| BC — Blanca Cuadrada | 16 |
| AP — Amarilla (cápsula) | 8 |
| RR — Rosada Redonda | 11 |
| AzC — Azul-Celeste | 6 |
| **TOTAL** | **54** |

---

## Estructura del repositorio

```
PDI-TP2/
├── Deteccion_pastillas.py   # pipeline principal
├── pills.png                # imagen de entrada
├── README.md
```

---

## Dependencias

```
opencv-python
numpy
matplotlib
```

Instalación:
```bash
pip install opencv-python numpy matplotlib
```

Ejecución:
```bash
python Deteccion_pastillas.py
```

---

## Decisiones de diseño

- **Sin valores hardcodeados de píxeles:** todos los umbrales se derivan de propiedades de la imagen (percentiles, proporciones relativas) o de geometría formal (circularidad ideal de círculo/cuadrado).
- **Sin deep learning ni OCR externo:** el pipeline usa únicamente OpenCV, NumPy y Matplotlib.
- **Clasificación determinística:** se priorizó la reproducibilidad sobre la generalidad, eligiendo reglas explícitas sobre clustering aleatorio.
- **CIELAB sobre RGB/HSV para color:** el desacople de luminosidad hace que las reglas cromáticas sean más robustas frente a cambios de iluminación.
