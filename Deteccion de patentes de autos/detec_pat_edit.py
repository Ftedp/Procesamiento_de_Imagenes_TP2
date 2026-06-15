import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CONSTANTES
# ==============================================================================

ESCALA_MAX = 800          # Lado máximo en píxeles tras redimensionar

# Filtro geométrico de candidatos
RATIO_MIN      = 2.0
RATIO_MAX      = 4.5
AREA_REL_MIN   = 0.005
AREA_REL_MAX   = 0.05

# Scoring: ratio
RATIO_IDEAL    = 3.08
UMBRAL_ANGULO  = 15       # grados; por encima se corrige con minAreaRect
DIST_RATIO_MAX = 0.5      # tolerancia alrededor del ratio ideal

# Scoring: píxeles blancos y negros
UMBRAL_BLANCO       = 200
UMBRAL_NEGRO        = 100
UMBRAL_SOLAP_BLANCO = 0.35
UMBRAL_SOLAP_NEGRO  = 0.10

# Scoring: contornos de caracteres dentro del candidato
RATIO_CAR_MIN   = 1.2     # hc/wc mínimo (más alto que ancho)
RATIO_CAR_MAX   = 2.5
AREA_CAR_MIN    = 0.04    # fracción del roi
AREA_CAR_MAX    = 0.20
N_CAR_MIN       = 3       # mínimo de contornos con forma de carácter
N_CAR_MAX       = 7       # máximo (la patente tiene 7 caracteres)

# Scoring: franja azul superior
UMBRAL_SOLAP_AZUL  = 0.25  # fracción mínima de píxeles azules en la franja superior
PROPORCION_FRANJA  = 0.20  # fracción del alto del candidato que se evalúa
# Rango HSV del azul de la patente Mercosur argentina
AZUL_HSV_BAJO  = np.array([100, 80, 50])
AZUL_HSV_ALTO  = np.array([130, 255, 255])

# Segmentación de caracteres (Parte B)
# RATIO_SEG_MIN  = 1.2
# RATIO_SEG_MAX  = 2.5
# AREA_SEG_MIN   = 0.03
# AREA_SEG_MAX   = 0.20
# prueba
RATIO_SEG_MIN = 1.0
RATIO_SEG_MAX = 4.0
AREA_SEG_MIN = 0.02
AREA_SEG_MAX = 0.25

# ==============================================================================
# FUNCIÓN 1: cargar y redimensionar
# ==============================================================================

def cargar_imagen(path):
    """
    Lee la imagen desde disco, la convierte a RGB y grises,
    y la redimensiona para que su lado más largo mida ESCALA_MAX píxeles.
    Retorna: img_bgr, img_rgb, img_gray (todas ya redimensionadas)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")

    h, w = img.shape[:2]
    escala = ESCALA_MAX / max(h, w)
    img = cv2.resize(img, (int(w * escala), int(h * escala)),
                     interpolation=cv2.INTER_AREA)

    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"  Shape redimensionada: {img.shape}  (escala={escala:.3f})")
    return img, img_rgb, img_gray


# ==============================================================================
# FUNCIÓN 2: preprocesar y detectar bordes
# ==============================================================================

def detectar_bordes(img_gray):
    """
    Aplica el pipeline blur→sharpen→blur→Canny sobre la imagen en grises.
    Retorna: edges (imagen binaria de bordes)
    """
    # Blur suave para no amplificar ruido con el sharpening
    img_suave = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sharpening: realza bordes restando el laplaciano local
    kernel_sharp = np.array([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=np.float32)
    img_sharp = cv2.filter2D(img_suave, -1, kernel_sharp)

    # Blur mayor antes de Canny para suprimir ruido fino post-sharpening
    img_blur = cv2.GaussianBlur(img_sharp, (5, 5), 0)
    edges = cv2.Canny(img_blur, 100, 200)

    return edges


# ==============================================================================
# FUNCIÓN 3: extraer candidatos (contornos con info geométrica)
# ==============================================================================

def extraer_candidatos(edges, img_area):
    """
    Encuentra contornos en la imagen de bordes y filtra por ratio y área relativa.
    Retorna: candidatos_sorted (lista de tuplas ordenadas por area_rel desc)
             cada tupla: (x, y, w, h, ratio, area_rel, cnt)
    """
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Total contornos encontrados: {len(contornos)}")

    candidatos_crudos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        ratio    = w / h
        area_rel = (w * h) / img_area
        candidatos_crudos.append((x, y, w, h, ratio, area_rel, cnt))

    # Filtro geométrico por ratio y área relativa
    candidatos_filtrados = [
        c for c in candidatos_crudos
        if RATIO_MIN < c[4] < RATIO_MAX and AREA_REL_MIN < c[5] < AREA_REL_MAX
    ]

    print(f"  Candidatos tras filtro ratio+área: {len(candidatos_filtrados)}")

    # Ordenados por area_rel descendente
    candidatos_sorted = sorted(candidatos_filtrados, key=lambda c: c[5],
                                reverse=True)
    return candidatos_sorted


# ==============================================================================
# FUNCIÓN 4: puntuar candidatos
# ==============================================================================

def puntuar_candidatos(candidatos_sorted, img_gray, img_bgr):
    """
    Aplica cuatro criterios de scoring sobre los candidatos y retorna
    la lista de puntajes (un entero por candidato, rango 0–4).

    Criterios:
      +1  ratio cercano a RATIO_IDEAL (con corrección por ángulo si > UMBRAL_ANGULO)
      +1  presencia simultánea de píxeles blancos Y negros en el ROI
      +1  entre N_CAR_MIN y N_CAR_MAX contornos con forma de carácter dentro del ROI
      +1  franja azul superior del candidato coincide con rango HSV del azul Mercosur
    """
    n = len(candidatos_sorted)
    puntajes = [0] * n

    img_h, img_w = img_gray.shape[:2]
    img_area = img_h * img_w

    # --- Criterio 1: ratio ---
    print(f"\n{'ID':>3} {'ratio_usado':>12} {'angulo':>8} {'dist_ratio':>11} {'puntos':>7}")
    print("-" * 45)
    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
        rect = cv2.minAreaRect(cnt)
        _, (rw, rh), angulo = rect
        if abs(angulo) > UMBRAL_ANGULO:
            ratio_usado = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
        else:
            ratio_usado = ratio
        dist_ratio = abs(ratio_usado - RATIO_IDEAL)
        if dist_ratio < DIST_RATIO_MAX:
            puntajes[i] += 1
        print(f"{i:>3} {ratio_usado:>12.2f} {angulo:>8.2f} {dist_ratio:>11.2f} {puntajes[i]:>7}")

    # --- Criterio 2: blanco Y negro ---
    mascara_blanco = (img_gray > UMBRAL_BLANCO).astype(np.uint8)
    mascara_negro  = (img_gray < UMBRAL_NEGRO).astype(np.uint8)

    print(f"\n{'ID':>3} {'solap_blanco':>13} {'solap_negro':>12} {'puntos':>7}")
    print("-" * 40)
    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
        roi_blanco   = mascara_blanco[y:y+h, x:x+w]
        roi_negro    = mascara_negro[y:y+h, x:x+w]
        solap_blanco = roi_blanco.sum() / (w * h)
        solap_negro  = roi_negro.sum()  / (w * h)
        if solap_blanco > UMBRAL_SOLAP_BLANCO and solap_negro > UMBRAL_SOLAP_NEGRO:
            puntajes[i] += 1
        print(f"{i:>3} {solap_blanco:>13.3f} {solap_negro:>12.3f} {puntajes[i]:>7}")

    # --- Criterio 3: contornos de caracteres ---
    print(f"\n{'ID':>3} {'contornos_car':>14} {'puntos':>7}")
    print("-" * 30)
    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
        roi      = img_gray[y:y+h, x:x+w]
        roi_area = w * h

        roi_bin = cv2.adaptiveThreshold(roi, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        11, 2)
        contornos_roi, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        n_car = 0
        for c in contornos_roi:
            xc, yc, wc, hc = cv2.boundingRect(c)
            if hc == 0 or wc == 0:
                continue
            area_rel_c = (wc * hc) / roi_area
            ratio_c    = hc / wc
            if RATIO_CAR_MIN < ratio_c < RATIO_CAR_MAX and \
               AREA_CAR_MIN  < area_rel_c < AREA_CAR_MAX:
                n_car += 1

        if N_CAR_MIN <= n_car <= N_CAR_MAX:
            puntajes[i] += 1
        print(f"{i:>3} {n_car:>14} {puntajes[i]:>7}")

    # --- Criterio 4: franja azul superior ---
    # Convertimos a HSV para detectar el azul característico de la patente Mercosur
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mascara_azul = cv2.inRange(img_hsv, AZUL_HSV_BAJO, AZUL_HSV_ALTO)

    print(f"\n{'ID':>3} {'solap_azul':>11} {'puntos':>7}")
    print("-" * 26)
    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos_sorted):
        # Evaluamos solo el 20% superior del bounding box
        h_franja = max(1, int(h * PROPORCION_FRANJA))
        roi_azul = mascara_azul[y:y+h_franja, x:x+w]
        solap_azul = roi_azul.sum() / (255 * w * h_franja)
        if solap_azul > UMBRAL_SOLAP_AZUL:
            puntajes[i] += 1
        print(f"{i:>3} {solap_azul:>11.3f} {puntajes[i]:>7}")

    return puntajes


# ==============================================================================
# FUNCIÓN 5: seleccionar ganador
# ==============================================================================

def seleccionar_ganador(candidatos_sorted, puntajes):
    """
    Elige el candidato con mayor puntaje.
    En caso de empate, desempata por posición vertical (mayor y = más abajo).
    Retorna: idx_ganador (índice en candidatos_sorted)
    """
    puntaje_max = max(puntajes)
    ganadores   = [i for i, p in enumerate(puntajes) if p == puntaje_max]

    if len(ganadores) == 1:
        idx_ganador = ganadores[0]
    else:
        # Desempate: el más abajo en la imagen (mayor y)
        idx_ganador = max(ganadores, key=lambda i: candidatos_sorted[i][1])

    print(f"\nPuntajes finales:")
    for i, p in enumerate(puntajes):
        marca = " <-- GANADOR" if i == idx_ganador else ""
        print(f"  ID {i}: {p} puntos{marca}")

    x, y, w, h, ratio, area_rel, cnt = candidatos_sorted[idx_ganador]
    print(f"Ganador: ID {idx_ganador}, pos=({x},{y}), size={w}x{h}, ratio={ratio:.2f}")

    return idx_ganador


# ==============================================================================
# FUNCIÓN 6: segmentar caracteres
# ==============================================================================

def segmentar_caracteres(crop, crop_gray):
    """
    Sobre el crop del candidato ganador, binariza y extrae contornos
    con forma de carácter.
    Retorna: caracteres_validos (lista de tuplas (xc, yc, wc, hc))
             crop_debug (imagen RGB con los bounding boxes dibujados)
    """
    roi_h, roi_w = crop_gray.shape[:2]
    roi_area = roi_h * roi_w

    # Probamos varios blockSizes y nos quedamos con el que da más cercano a 7 caracteres
    mejores_caracteres = []
    mejor_diff = 999
    mejor_bin  = None

    for bs in [7, 9, 11, 13, 15]:
        bin_test = cv2.adaptiveThreshold(crop_gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         bs, 2)
        conts_test, _ = cv2.findContours(bin_test, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        chars_test = []
        for c in conts_test:
            xc, yc, wc, hc = cv2.boundingRect(c)
            if hc == 0 or wc == 0:
                continue
            area_rel_c = (wc * hc) / roi_area
            ratio_c    = hc / wc
            if RATIO_SEG_MIN < ratio_c < RATIO_SEG_MAX and \
               AREA_SEG_MIN  < area_rel_c < AREA_SEG_MAX:
                chars_test.append((xc, yc, wc, hc))

        diff = abs(len(chars_test) - 7)
        if diff < mejor_diff:
            mejor_diff = diff
            mejores_caracteres = chars_test
            mejor_bin  = bin_test

    caracteres_validos = mejores_caracteres
    print(f"  Caracteres válidos: {len(caracteres_validos)}")

    crop_debug = crop.copy()
    for xc, yc, wc, hc in caracteres_validos:
        cv2.rectangle(crop_debug, (xc, yc), (xc+wc, yc+hc), (0, 255, 0), 1)

    return caracteres_validos, crop_debug



# ============== funcion extraer candidatos alternativo sin comentarios ======================================
# def extraer_candidatos_alternativo(img_gray, img_area):

#     kernel_bh = cv2.getStructuringElement( cv2.MORPH_RECT, (13, 5))

#     blackhat = cv2.morphologyEx( img_gray, cv2.MORPH_BLACKHAT, kernel_bh)

#     grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)

#     grad_x = np.absolute(grad_x)

#     grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

#     grad_x = cv2.GaussianBlur( grad_x, (5,5), 0)

#     _, thresh = cv2.threshold( grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17,5))

#     thresh = cv2.morphologyEx( thresh, cv2.MORPH_CLOSE, kernel_close)

#     thresh = cv2.dilate( thresh, None, iterations=2)

#     contornos, _ = cv2.findContours( thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     candidatos = []

#     for cnt in contornos:

#         x, y, w, h = cv2.boundingRect(cnt)

#         if h == 0:
#             continue

#         ratio = w / h

#         area_rel = (w * h) / img_area

#         if (2.0 < ratio < 6.0 and 0.003 < area_rel < 0.08):
            
#             candidatos.append((x,y,w,h,ratio,area_rel,cnt))

#     candidatos = sorted(
#         candidatos,
#         key=lambda c: c[5],
#         reverse=True
#     )

#     print(
#         f"  Candidatos alternativos: "
#         f"{len(candidatos)}"
#     )

#     return candidatos


# misma funcionque la anterior con comentarios detallados explicando cada paso del pipeline 
def extraer_candidatos_alternativo(img_gray, img_area):
    """
    Pipeline alternativo para la detección de la ROI de la patente.
    Combina operadores morfológicos (U6) y detección de bordes por gradiente (U3)
    para aislar regiones con alta densidad de transiciones verticales.
    """
    
    # 1. Definición del Elemento Estructural (SE) rectangular y asimétrico (Unidad 6)
    # Al ser más ancho que alto (13x5), se adapta al formato alargado de la patente.
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

    # 2. Operación Morfológica de BlackHat (Unidad 6)
    # BlackHat = Clausura(Img) - Img. Resalta elementos oscuros sobre fondos claros, 
    # ideal para aislar los caracteres negros de la patente sobre la placa blanca.
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # 3. Operador de Derivada Espacial Sobel en el eje X (Unidad 3 - Segmentación por Bordes)
    # Calcula la primera derivada horizontal (dx=1, dy=0). Como los caracteres de la patente 
    # generan muchos cambios bruscos de intensidad de izquierda a derecha, SobelX produce un 
    # patrón denso de bordes verticales de alta frecuencia espaciados muy cerca entre sí.
    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)

    # 4. Cálculo del Valor Absoluto (Unidad 3)
    # Las transiciones de blanco a negro dan derivadas negativas y de negro a blanco positivas.
    # El valor absoluto unifica ambos bordes en valores positivos de magnitud de gradiente.
    grad_x = np.absolute(grad_x)

    # 5. Normalización y Casteo de Tipo de Dato (Unidad 1 - Fundamentos)
    # Escala linealmente el rango dinámico de las amplitudes de gradiente a [0, 255].
    # Luego se convierte a uint8 (enteros de 8 bits sin signo) requerido por las funciones de OpenCV.
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 6. Suavizado mediante Filtro Gaussiano Lineal (Unidad 2 - Filtrado Espacial)
    # Un filtro de media pesada pasabajo con máscara de 5x5. Borra variaciones de ruido de alta
    # frecuencia y ayuda a que los bordes individuales de los caracteres comiencen a "difundirse"
    # y conectarse entre sí espaciadamente en una sola masa difusa regional.
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)

    # 7. Segmentación por Umbralado Global usando el Método de Otsu (Unidad 3 - Segmentación)
    # Binariza la imagen calculando automáticamente el umbral 'T' óptimo que maximiza la varianza 
    # interclase del histograma bimodal resultante, separando las zonas con mucho texto (blanco) del resto.
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. Operación Morfológica de Clausura (Unidad 6)
    # Clausura = Dilatación seguida de Erosión. Al aplicar un elemento estructural horizontal (17x5),
    # conecta y fusiona todos los caracteres binarizados individuales de la patente en un único bloque 
    # rectangular continuo y sólido, cerrando los gaps (huecos) inter-caracteres.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # 9. Operación de Dilatación Morfológica (Unidad 6)
    # Aplica 2 iteraciones de dilatación pura para "engrosar" las regiones blancas binarizadas.
    # Asegura la continuidad de los bordes externos y rellena cualquier microporo residual.
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 10. Extracción de Contornos Externos (Unidad 3)
    # RETR_EXTERNAL recupera únicamente las siluetas exteriores del bloque (descarta huecos internos).
    # CHAIN_APPROX_SIMPLE comprime los segmentos horizontales, verticales y diagonales redundantes.
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []

    # 11. Filtrado Geométrico de Candidatos (Unidad 3 - Descriptores de Forma)
    for cnt in contornos:
        # Obtiene la caja contenedora alineada a los ejes (Bounding Box)
        x, y, w, h = cv2.boundingRect(cnt)

        # Control de seguridad matemática para evitar indeterminaciones (división por cero)
        if h == 0:
            continue

        # Calcula la Relación de Aspecto (Aspect Ratio = ancho / alto)
        # La patente Mercosur/Argentina ideal posee un valor de aprox. 3.08 (400mm / 130mm)
        ratio = w / h

        # Calcula el Área Relativa de la componente respecto al área total de la imagen
        area_rel = (w * h) / img_area

        # Filtro de Clasificación Geométrica basado en el conocimiento a priori de la placa:
        # Filtra que la relación de aspecto sea horizontal alargada (2.0 < ratio < 6.0)
        # y que el objeto no sea ni gigante (carrocería) ni minúsculo (ruido): (0.003 < area_rel < 0.08)
        if (2.0 < ratio < 6.0 and 0.003 < area_rel < 0.08):
            # Guardamos las estadísticas y el contorno del candidato que superó la condición
            candidatos.append((x, y, w, h, ratio, area_rel, cnt))

    # 12. Ordenamiento de Datos (Lógica de NumPy / Python)
    # Ordena la lista de mayor a menor basándose en el descriptor de Área Relativa (índice 5).
    # Asume de forma heurística que la patente será una de las componentes válidas de mayor tamaño.
    candidatos = sorted(
        candidatos,
        key=lambda c: c[5],
        reverse=True
    )

    print(
        f"  Candidatos alternativos: "
        f"{len(candidatos)}"
    )

    return candidatos





# ==============================================================================
# FUNCIÓN 7: procesar una imagen completa
# ==============================================================================

# def procesar_imagen(path, mostrar=False):
#     """
#     Ejecuta el pipeline completo sobre una imagen:
#       1. Cargar y redimensionar
#       2. Detectar bordes
#       3. Extraer y filtrar candidatos
#       4. Puntuar candidatos
#       5. Seleccionar ganador
#       6. Segmentar caracteres

#     Retorna: crop_ganador (RGB), caracteres_validos, puntaje_max
#              o (None, [], 0) si no hay candidatos.
#     """
#     print(f"\n{'='*60}")
#     print(f"Procesando: {path}")
#     print(f"{'='*60}")

#     # 1. Cargar
#     img_bgr, img_rgb, img_gray = cargar_imagen(path)
#     img_h, img_w = img_gray.shape[:2]
#     img_area = img_h * img_w

#     # 2. Bordes
#     edges = detectar_bordes(img_gray)

#     # 3. Candidatos
#     candidatos_sorted = extraer_candidatos(edges, img_area)

#     if not candidatos_sorted:
#         print("  Sin candidatos tras el filtro. Imagen no procesada.")
#         return None, [], 0

#     # 4. Puntuar (se pasa img_bgr para el criterio de azul)
#     puntajes = puntuar_candidatos(candidatos_sorted, img_gray, img_bgr)

#     # 5. Ganador
#     idx_ganador = seleccionar_ganador(candidatos_sorted, puntajes)
#     puntaje_max = max(puntajes)

#     x, y, w, h, ratio, area_rel, cnt = candidatos_sorted[idx_ganador]
#     crop_ganador  = img_rgb[y:y+h, x:x+w]
#     crop_gray_win = img_gray[y:y+h, x:x+w]

#     # 6. Segmentar
#     caracteres_validos, crop_debug = segmentar_caracteres(crop_ganador,
#                                                           crop_gray_win)

#     # Visualización opcional
#     if mostrar:
#         fig, axs = plt.subplots(1, 3, figsize=(14, 4))
#         axs[0].imshow(img_rgb)
#         axs[0].set_title('Original')
#         axs[0].axis('off')
#         axs[1].imshow(crop_ganador)
#         axs[1].set_title(f'Ganador (ratio={ratio:.2f}, pts={puntaje_max})')
#         axs[1].axis('off')
#         axs[2].imshow(crop_debug)
#         axs[2].set_title(f'Caracteres ({len(caracteres_validos)})')
#         axs[2].axis('off')
#         fig.suptitle(path)
#         plt.tight_layout()
#         plt.show(block=False)

#     return crop_ganador, caracteres_validos, puntaje_max

# ========================== prueba con filtro alternativo =======================================

def procesar_imagen(path, mostrar=False):
    """
    Ejecuta el pipeline completo sobre una imagen:
      1. Cargar y redimensionar
      2. Detectar bordes
      3. Extraer y filtrar candidatos
      4. Puntuar candidatos
      5. Seleccionar ganador
      6. Segmentar caracteres

    Retorna: crop_ganador (RGB), caracteres_validos, puntaje_max
             o (None, [], 0) si no hay candidatos.
    """
    print(f"\n{'='*60}")
    print(f"Procesando: {path}")
    print(f"{'='*60}")

    # 1. Cargar
    img_bgr, img_rgb, img_gray = cargar_imagen(path)
    img_h, img_w = img_gray.shape[:2]
    img_area = img_h * img_w

    # 2. Bordes
    edges = detectar_bordes(img_gray)

    # 3. Candidatos
    candidatos_sorted = extraer_candidatos(edges, img_area)

    if not candidatos_sorted:
        print("  Sin candidatos tras el filtro. Imagen no procesada.")
        return None, [], 0

    # 4. Puntuar (se pasa img_bgr para el criterio de azul)
    puntajes = puntuar_candidatos(candidatos_sorted, img_gray, img_bgr)

    # 5. Ganador
    idx_ganador = seleccionar_ganador(candidatos_sorted, puntajes)
    puntaje_max = max(puntajes)

    # Prueba de filtro alternativo si el puntaje es bajo
    if puntaje_max < 2:

        print("\n>>> Puntaje bajo.", "\n>>> Ejecutando detector alternativo...")

        candidatos_alt = extraer_candidatos_alternativo( img_gray, img_area)

        if candidatos_alt:

            puntajes_alt = puntuar_candidatos( candidatos_alt, img_gray, img_bgr)

            if max(puntajes_alt) > puntaje_max:

                print(
                    ">>> Detector alternativo mejoró "
                    "el resultado."
                )

                candidatos_sorted = candidatos_alt

                puntajes = puntajes_alt

                idx_ganador = seleccionar_ganador( candidatos_sorted, puntajes)

                puntaje_max = max(puntajes)

    x, y, w, h, ratio, area_rel, cnt = candidatos_sorted[idx_ganador]
    crop_ganador  = img_rgb[y:y+h, x:x+w]
    crop_gray_win = img_gray[y:y+h, x:x+w]

    # 6. Segmentar
    caracteres_validos, crop_debug = segmentar_caracteres(crop_ganador,
                                                          crop_gray_win)

    # Visualización opcional
    if mostrar:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].imshow(img_rgb)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(crop_ganador)
        axs[1].set_title(f'Ganador (ratio={ratio:.2f}, pts={puntaje_max})')
        axs[1].axis('off')
        axs[2].imshow(crop_debug)
        axs[2].set_title(f'Caracteres ({len(caracteres_validos)})')
        axs[2].axis('off')
        fig.suptitle(path)
        plt.tight_layout()
        plt.show(block=False)

    return crop_ganador, caracteres_validos, puntaje_max


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':

    resultados = {}

    for n in range(1, 13):
        #path = f'img_{n}.jpg'
        path = f'img_{n}.jpg'
        try:
            crop, caracteres, puntaje = procesar_imagen(path, mostrar=True)
            resultados[path] = {
                'exito':      crop is not None,
                'puntaje':    puntaje,
                'caracteres': len(caracteres) if caracteres else 0,
            }
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            resultados[path] = {'exito': False, 'puntaje': 0, 'caracteres': 0}

    # Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"{'Imagen':<12} {'Éxito':>6} {'Puntaje':>8} {'Caracteres':>11}")
    print("-" * 40)
    exitos = 0
    for path, r in resultados.items():
        marca = 'SI' if r['exito'] else 'NO'
        if r['caracteres'] == 7:
            exitos += 1
        print(f"{path:<12} {marca:>6} {r['puntaje']:>8} {r['caracteres']:>11}")
    print(f"\nDetectadas correctamente: {exitos}/12")

    plt.show()