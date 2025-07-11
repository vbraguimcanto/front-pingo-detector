import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from pathlib import Path

# ---------- Estilo ----------
st.set_page_config(page_title="Detecção de Pingos", page_icon="🔎", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        background-color: #eafaf1;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Detecção de Pingos")
st.write("Mostra imagem original e bounding box final.")

# ---------- Carregar modelo ----------
@st.cache_resource
def load_model():
    model_path = Path("best.pt")
    if not model_path.exists():
        st.error("❌ Arquivo 'best.pt' não encontrado no diretório atual.")
        st.stop()
    return YOLO(str(model_path))

model = load_model()

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    CONF_THRESHOLD = 0.4
    largest_box_yolo = None

    # ---------- YOLO detection ----------
    with st.spinner("IA Detectando..."):
        results = model(image_bgr)
        result = results[0]
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if largest_box_yolo is None or area > (largest_box_yolo[2] - largest_box_yolo[0]) * (largest_box_yolo[3] - largest_box_yolo[1]):
                    largest_box_yolo = (x1, y1, x2, y2)

    box_to_draw = None

    # ---------- Se YOLO detectou algo válido, comparar com OpenCV ----------
    if largest_box_yolo:
        with st.spinner("Detectando com algoritmo tradicional..."):
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_area = image_bgr.shape[0] * image_bgr.shape[1]
            largest_box_cv = None
            largest_area_cv = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if 1000 < area < 0.5 * image_area and area > largest_area_cv:
                    largest_area_cv = area
                    largest_box_cv = (x, y, x + w, y + h)

        if largest_box_cv:
            iou = calculate_iou(largest_box_yolo, largest_box_cv)
            box_to_draw = largest_box_yolo if iou > 0.5 else largest_box_cv
        else:
            box_to_draw = largest_box_yolo
    # ---------- Se YOLO não detectou nada >= 50%, não plota nada ----------

    # ---------- Desenho final ----------
    annotated_img = image_bgr.copy()
    if box_to_draw:
        x1, y1, x2, y2 = box_to_draw
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde, sem texto

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Imagem Original")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("✅ Resultado final")
        st.image(annotated_pil, use_container_width=True)

    buffered = BytesIO()
    annotated_pil.save(buffered, format="PNG")
    st.download_button(
        label="📥 Baixar imagem anotada",
        data=buffered.getvalue(),
        file_name="resultado_hibrido.png",
        mime="image/png"
    )

else:
    st.info("Por favor, envie uma imagem no campo acima.")

st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Desenvolvido por Quorion
</p>
""", unsafe_allow_html=True)
