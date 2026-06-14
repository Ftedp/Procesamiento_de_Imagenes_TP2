import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def imshow(img, title=None, color_img=True, cmap='gray'):
    """Muestra una imagen con matplotlib."""
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show(block=False)


def detectar_roi(img_gray):
    """
    Detecta automáticamente los límites verticales de la cinta transportadora.

    Estrategia:
    - Calcula el perfil de intensidad promedio por fila.
    - Suaviza el perfil con un promedio móvil para eliminar variaciones pequeñas.
    - Calcula la derivada del perfil suavizado.
    - El límite superior es la caída más brusca en la primera mitad (metal → cinta oscura).
    - El límite inferior es la subida más brusca en la segunda mitad (cinta oscura → metal).

    Parámetros
    ----------
    img_gray : np.ndarray
        Imagen en escala de grises.

    Retorna
    -------
    fila_superior, fila_inferior : int, int
        Filas que delimitan la ROI de la cinta.
    """
    perfil = np.mean(img_gray, axis=1)
    perfil_suavizado = np.convolve(perfil, np.ones(20) / 20, mode='same')
    derivada = np.diff(perfil_suavizado, prepend=perfil_suavizado[0])

    mitad = len(derivada) // 2
    fila_superior = np.argmin(derivada[:mitad])
    fila_inferior = mitad + np.argmax(derivada[mitad:])

    return fila_superior, fila_inferior


def segmentar_pastillas(roi, roi_gray):
    """
    Segmenta las pastillas del fondo oscuro de la cinta.

    Estrategia:
    - Convierte la ROI a HSV y extrae el canal V (brillo).
    - Umbraliza el canal V: las pastillas tienen V alto, el fondo tiene V bajo.
      El umbral se elige en el valle del histograma de V (~110).
    - Aplica cierre morfológico para rellenar huecos pequeños dentro de las pastillas.
    - Detecta contornos externos.
    - Filtra contornos por área mínima (0.14% del área de la ROI) y relación de aspecto
      para eliminar ruido y bordes de la ROI.

    Parámetros
    ----------
    roi : np.ndarray
        ROI en color (RGB).
    roi_gray : np.ndarray
        ROI en escala de grises.

    Retorna
    -------
    contornos_filtrados : list
        Lista de contornos válidos, uno por pastilla.
    """
    # Umbralización sobre canal V
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    V = roi_hsv[:, :, 2]
    
    # MODIFICACION ----------####---------------------- 
    umbral = np.percentile(V, 88)  # umbral dinámico basado en el percentil 88 del canal V

    _, mask = cv2.threshold(V, umbral, 255, cv2.THRESH_BINARY)

    # Cierre morfológico para rellenar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detección de contornos
    contornos, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtro por área mínima (0.14% del área de la ROI) y relación de aspecto
    area_roi = roi.shape[0] * roi.shape[1]
    area_minima = 0.0014 * area_roi  # una pastilla ocupa al menos el 0.14% de la cinta

    #Filtrado por area minima y por ratio.
    contornos_filtrados = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_minima:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h
        if aspect_ratio > 10:  # contornos muy anchos son bordes de la ROI, no pastillas
            continue
        contornos_filtrados.append(c)

    return contornos_filtrados


# ----------------------- funciones con kmeans para clasificar por color y forma -----------------------
#def identificar_grupo_color(centers):
    """
    Identifica a qué tipo de pastilla corresponde cada grupo del K-means,
    basándose en las propiedades de los centros en espacio LAB.

    Propiedades usadas:
    - Blancas:   L muy alto (~230), A y B neutros (~128)
    - Amarillas: B alto (>128, componente amarillo-azul de LAB)
    - Rosadas:   A alto (>128, componente rojo-verde de LAB)
    - Azules:    B bajo (<128)

    Parámetros
    ----------
    centers : np.ndarray
        Centros LAB de cada grupo K-means, shape (K, 3).

    Retorna
    -------
    grupo_a_tipo : dict
        Diccionario {índice_grupo: tipo} donde tipo es 'AP', 'RR', 'blanca' o 'AzC'.
    """
    grupo_a_tipo = {}
    for k, centro in enumerate(centers):
        L, A, B = centro
        if L > 200:             # L muy alto → blancas (incluye amarillas claras)
            if B > 140:         # B alto dentro de las claras → amarillas
                grupo_a_tipo[k] = 'AP'
            else:
                grupo_a_tipo[k] = 'blanca'
        elif A > 140:           # A alto → rosadas
            grupo_a_tipo[k] = 'RR'
        else:                   # B bajo → azules
            grupo_a_tipo[k] = 'AzC'
    return grupo_a_tipo

#def clasificar_por_color(contornos_filtrados, roi):
    """
    Clasifica cada pastilla por color usando K-means en espacio LAB.

    Estrategia:
    - Extrae el color promedio LAB de cada pastilla usando su contorno como máscara.
    - Aplica K-means con K=4 sobre esos colores.
    - Identifica automáticamente el tipo de cada grupo mirando los centros LAB.

    Parámetros
    ----------
    contornos_filtrados : list
        Lista de contornos válidos.
    roi : np.ndarray
        ROI en color (RGB).

    Retorna
    -------
    labels : np.ndarray
        Etiqueta de grupo K-means para cada pastilla.
    grupo_a_tipo : dict
        Mapeo de índice de grupo a tipo de pastilla.
    """
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)

    # Extraigo color promedio LAB de cada pastilla
    colores_promedio = []
    for c in contornos_filtrados:
        mask_pastilla = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_pastilla, [c], -1, 255, -1)
        color_medio = cv2.mean(roi_lab, mask=mask_pastilla)[:3]
        colores_promedio.append(color_medio)

    colores_promedio = np.float32(colores_promedio)

    # K-means con K=4
    K = 4
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(colores_promedio, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Identifico automáticamente qué grupo es qué color
    grupo_a_tipo = identificar_grupo_color(centers)

    return labels, grupo_a_tipo

#def clasificar_blancas_por_forma(contornos_filtrados, labels, grupo_a_tipo):
    """
    Subdivide las pastillas blancas en redondas (BR) y cuadradas (BC)
    usando K-means sobre la circularidad de cada contorno.

    Circularidad = 4 * pi * area / perimetro^2
    - Círculo perfecto → 1.0
    - Cuadrado         → ~0.785
    El grupo con centro de circularidad más alto son las redondas (BR),
    el de centro más bajo son las cuadradas (BC).

    Parámetros
    ----------
    contornos_filtrados : list
        Lista de contornos válidos.
    labels : np.ndarray
        Etiqueta de grupo K-means de color para cada pastilla.
    grupo_a_tipo : dict
        Mapeo de índice de grupo a tipo.

    Retorna
    -------
    labels_forma : dict
        Diccionario {índice_en_contornos_filtrados: tipo_final}
        donde tipo_final es 'BR' o 'BC' para las blancas.
    """
    # Obtengo índices de las pastillas blancas
    grupo_blanco = [k for k, v in grupo_a_tipo.items() if v == 'blanca'][0]
    indices_blancas = np.where(labels.flatten() == grupo_blanco)[0]

    # Calculo circularidad de cada blanca
    circularidades = []
    for i in indices_blancas:
        c = contornos_filtrados[i]
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        circularidad = 4 * np.pi * area / (perimetro ** 2)
        circularidades.append([circularidad])

    circularidades = np.float32(circularidades)

    # K-means con K=2 sobre circularidad
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels_circ, centers_circ = cv2.kmeans(circularidades, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # El grupo con centro más alto es el de las redondas
    idx_redondas = 0 if centers_circ[0] > centers_circ[1] else 1
    idx_cuadradas = 1 - idx_redondas

    # Armo diccionario índice → tipo final
    labels_forma = {}
    for j, i in enumerate(indices_blancas):
        if labels_circ[j] == idx_redondas:
            labels_forma[i] = 'BR'
        else:
            labels_forma[i] = 'BC'

    return labels_forma

# ----------------------- funciones sin kmeans para clasificar por color y forma -----------------------
#  se reescriben: identificar_grupo_color(), clasificar_por_color() ,clasificar_blancas_por_forma()
def identificar_grupo_color(roi, contorno):
    """
    Analiza una única pastilla mediante su contorno usando CIELAB y HSV.
    Ignora la iluminación (canal L) y clasifica basándose en los ejes cromáticos.

    - 'RR' (Rojo-Rosado): Canal A alto (positivo, > 128)
    - 'AP' (Amarillo): Canal B alto (positivo, > 128)
    - 'AzC' (Azul-Celeste): Canal B bajo (negativo, < 128). Al ser mitad blanca 
                            y mitad azul, el promedio de B baja de 128.
    - 'blanca': Canales A y B cercanos al neutro (128) y alto brillo en HSV.
    """
    # 1. Crear máscara para aislar la pastilla del fondo
    mask_pastilla = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_pastilla, [contorno], -1, 255, -1)

    # 2. Convertir la ROI a LAB y HSV 
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    
    # 3. Extraer promedios ignorando el fondo
    media_lab = cv2.mean(roi_lab, mask=mask_pastilla)
    media_hsv = cv2.mean(roi_hsv, mask=mask_pastilla)
    
    # 4. Extraer canales (OpenCV ubica el neutro en 128 para uint8)
    canal_a = media_lab[1] # Verde < 128 < Rojo/Rosado
    canal_b = media_lab[2] # Azul < 128 < Amarillo
    brillo_v = media_hsv[2]
    saturacion_s = media_hsv[1]
    # --- LÓGICA DE UMBRALADO CROMÁTICO PÚRO ---
    if canal_a > 140:
        # Fuerte componente rojiza
        return 'RR'
        
    elif canal_b > 138:
        # Fuerte componente amarilla
        return 'AP'
        
    elif canal_b < 120:
        # Fuerte componente azulada (tira el promedio por debajo de 128)
        return 'AzC'
        
    else:
        # Si A y B son neutros, es un color acromático. Validamos que sea blanca 
        # (alto brillo, baja saturación) para no confundir con sombras.
        if brillo_v > 120 and saturacion_s < 60:
            return 'blanca'
        else:
            # Color no reconocido o sombra
            return 'indefinido'


def clasificar_por_color(contornos_filtrados, roi):
    """
    Clasifica cada pastilla por color de forma determinística.

    Parámetros
    ----------
    contornos_filtrados : list
        Lista de contornos válidos.
    roi : np.ndarray
        ROI en color (RGB).

    Retorna
    -------
    labels : np.ndarray
        Arreglo de etiquetas (shape: N x 1) para cada pastilla.
    grupo_a_tipo : dict
        Mapeo de índice de grupo a tipo de pastilla.
    """
    # Mapeo estático que reemplaza los centros aleatorios de K-means
    grupo_a_tipo = {
        0: 'AP',
        1: 'RR',
        2: 'blanca',
        3: 'AzC',
        -1: 'indefinido'
    }
    
    # Invertimos el diccionario para buscar el ID fácilmente
    tipo_a_grupo = {v: k for k, v in grupo_a_tipo.items()}
    
    lista_etiquetas = []
    
    # Evaluamos cada pastilla una por una
    for contorno in contornos_filtrados:
        color_detectado = identificar_grupo_color(roi, contorno)
        id_grupo = tipo_a_grupo[color_detectado]
        lista_etiquetas.append(id_grupo)
        
    # K-means original devuelve los labels con shape (N, 1) y tipo int32.
    # Replicamos exactamente esa estructura.
    labels = np.array(lista_etiquetas, dtype=np.int32).reshape(-1, 1)
    
    return labels, grupo_a_tipo

def clasificar_blancas_por_forma(contornos_filtrados, labels, grupo_a_tipo):
    """
    Subdivide las pastillas blancas en redondas (BR) y cuadradas (BC)
    usando un umbral estático de circularidad geométrica.
    """
    # Buscamos cuál es el ID numérico asignado a las pastillas blancas
    grupo_blanco = [k for k, v in grupo_a_tipo.items() if v == 'blanca'][0]
    indices_blancas = np.where(labels.flatten() == grupo_blanco)[0]

    labels_forma = {}

    for i in indices_blancas:
        c = contornos_filtrados[i]
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)

        # # Prevención de división por cero
        if perimetro == 0:
            circularidad = 0
        else:
            circularidad = 4 * np.pi * area / (perimetro ** 2)

        # # Clasificación basada en un umbral de circularidad
        # # Círculo ideal = 1.0, Cuadrado ideal = 0.785

        if circularidad > 0.88:
            labels_forma[i] = 'BR'  # Blanca Redonda
        else:
            labels_forma[i] = 'BC'  # Blanca Cuadrada

    return labels_forma


def asignar_etiquetas(contornos_filtrados, labels, grupo_a_tipo, labels_forma):
    """
    Asigna tipo e ID a cada pastilla combinando la clasificación por color y por forma.

    Parámetros
    ----------
    contornos_filtrados : list
    labels : np.ndarray
    grupo_a_tipo : dict
    labels_forma : dict

    Retorna
    -------
    etiquetas : list
        Lista de tuplas (contorno, tipo, id_pastilla).
    conteos : dict
        Conteo total por tipo.
    """
    conteos = {'BR': 0, 'BC': 0, 'AP': 0, 'RR': 0, 'AzC': 0}
    etiquetas = []

    for i, c in enumerate(contornos_filtrados):
        grupo_color = labels.flatten()[i]
        tipo_color = grupo_a_tipo[grupo_color]

        if tipo_color == 'blanca':
            tipo = labels_forma[i]  # BR o BC
        else:
            tipo = tipo_color       # AP, RR o AzC

        conteos[tipo] += 1
        etiquetas.append((c, tipo, conteos[tipo]))

    return etiquetas, conteos


def generar_imagen_resultado(roi, etiquetas):
    """
    Genera la imagen final con contornos y etiquetas sobre cada pastilla.

    Parámetros
    ----------
    roi : np.ndarray
        ROI en color (RGB).
    etiquetas : list
        Lista de tuplas (contorno, tipo, id_pastilla).

    Retorna
    -------
    img_resultado : np.ndarray
        Imagen con contornos y etiquetas dibujadas.
    """
    img_resultado = roi.copy()

    for c, tipo, id_pastilla in etiquetas:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.drawContours(img_resultado, [c], -1, (0, 255, 0), 2)

        etiqueta = f"{tipo}{id_pastilla}"
        cv2.putText(img_resultado, etiqueta, (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img_resultado


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

if __name__ == '__main__':

    # --- A: Cargar imagen y segmentar ROI ------------------------------------
    img = cv2.imread('pills.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fila_superior, fila_inferior = detectar_roi(img_gray)
    print(f"[A] ROI detectada: filas {fila_superior} a {fila_inferior}")

    roi = img_rgb[fila_superior:fila_inferior, :]
    roi_gray = img_gray[fila_superior:fila_inferior, :]
    imshow(roi, title=f'A - ROI de la cinta transportadora')

    # --- B: Detectar y segmentar cada pastilla -------------------------------
    contornos_filtrados = segmentar_pastillas(roi, roi_gray)
    print(f"[B] Pastillas detectadas: {len(contornos_filtrados)}")

    roi_contornos = roi.copy()
    cv2.drawContours(roi_contornos, contornos_filtrados, -1, (0, 255, 0), 2)
    imshow(roi_contornos, title=f'B - Pastillas segmentadas: {len(contornos_filtrados)}')

    # --- C: Clasificar por color y forma -------------------------------------
    labels, grupo_a_tipo = clasificar_por_color(contornos_filtrados, roi)
    print(f"[C] Grupos de color identificados: {grupo_a_tipo}")

    labels_forma = clasificar_blancas_por_forma(contornos_filtrados, labels, grupo_a_tipo)

    etiquetas, conteos = asignar_etiquetas(contornos_filtrados, labels, grupo_a_tipo, labels_forma)

    # --- D: Reporte por consola y imagen resultado ---------------------------
    print("\n=== RESULTADOS DE CLASIFICACIÓN ===")
    for tipo, n in conteos.items():
        print(f"  {tipo}: {n} pastillas")
    print(f"  TOTAL: {sum(conteos.values())} pastillas")

    img_resultado = generar_imagen_resultado(roi, etiquetas)
    imshow(img_resultado, title='D - Resultado final')

    plt.show()