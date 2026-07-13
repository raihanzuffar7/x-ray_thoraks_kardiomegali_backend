import io
import os
import base64
import requests

import numpy as np
import streamlit as st
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Konfigurasi halaman
# --------------------------------------------------------------------------
st.set_page_config(page_title="YOLOv8s vs RetinaNet", layout="wide")
st.title("Bandingkan Model: YOLOv8s vs RetinaNet")

tab1, tab2 = st.tabs(["🖥️ Test Langsung (Local)", "🌐 Test via FastAPI"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Dipakai hanya kalau auto-detect dimatikan/gagal menebak jumlah class dari checkpoint
DEFAULT_RETINA_NUM_CLASSES = 91

# --------------------------------------------------------------------------
# Path model lokal — sesuaikan dengan lokasi file weight kamu
# --------------------------------------------------------------------------
YOLO_MODEL_PATH = "../runs/yolov8s_kardio/weights/best.pt"
RETINA_MODEL_PATH = "../outputs/retinanet_kardio_best.pt"


# --------------------------------------------------------------------------
# Helper umum
# --------------------------------------------------------------------------
def get_color(label: int):
    """Warna konsisten per class id (deterministik dari hash)."""
    rng = np.random.default_rng(label * 9973 + 17)
    return tuple(int(x) for x in rng.integers(50, 230, size=3))


def resize_image_if_needed(image: Image.Image, max_dim: int) -> Image.Image:
    """Perkecil gambar jika sisi terpanjang melebihi max_dim (mempercepat inferensi di CPU)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS)


def get_scaled_font(image_size: tuple, scale_factor: float = 0.03):
    """Load font dengan ukuran proporsional terhadap dimensi gambar,
    supaya label tidak kelihatan mini di gambar beresolusi besar.
    Dijamin selalu mengembalikan font yang valid (fallback berlapis)."""
    font_size = max(16, int(min(image_size) * scale_factor))
    for font_name in ("arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=font_size)  # Pillow >= 10.1
    except Exception:
        pass
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# Loader YOLOv8s (ultralytics)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model YOLOv8s...")
def load_yolo_model(path: str):
    from ultralytics import YOLO

    model = YOLO(path)
    return model


def run_yolo_inference(model, image: Image.Image, conf: float, iou: float):
    results = model.predict(source=np.array(image), conf=conf, iou=iou, verbose=False)
    r = results[0]

    annotated_bgr = r.plot()  # numpy array, BGR
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_img = Image.fromarray(annotated_rgb)

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
        score = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({"class": cls_name, "score": round(score, 3), "box": [round(v, 1) for v in xyxy]})

    return annotated_img, detections


# --------------------------------------------------------------------------
# Loader RetinaNet (torchvision)
# --------------------------------------------------------------------------
def extract_state_dict(checkpoint):
    """Handle beberapa format checkpoint umum: state_dict langsung, atau
    dibungkus dalam dict dengan key "model"/"state_dict"."""
    if isinstance(checkpoint, dict) and "model" in checkpoint and not any(
        k.startswith(("backbone", "head")) for k in checkpoint.keys()
    ):
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def infer_num_classes_from_state_dict(state_dict, num_anchors: int = 9):
    """Tebak jumlah class (termasuk background) dari shape layer cls_logits.
    Asumsi default torchvision: 9 anchor per titik."""
    key = "head.classification_head.cls_logits.weight"
    if key in state_dict:
        out_channels = state_dict[key].shape[0]
        if out_channels % num_anchors == 0:
            return out_channels // num_anchors
    return None


def build_retinanet(variant: str, num_classes: int):
    if variant == "retinanet_resnet50_fpn_v2":
        return torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=None, weights_backbone=None, num_classes=num_classes
        )
    return torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None, weights_backbone=None, num_classes=num_classes
    )


@st.cache_resource(show_spinner="Loading model RetinaNet...")
def load_retinanet_model(
    path: str,
    num_classes_override: int,
    backbone_variant: str,
    auto_detect: bool,
):
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)

    detected_num_classes = infer_num_classes_from_state_dict(state_dict)
    num_classes = detected_num_classes if (auto_detect and detected_num_classes) else num_classes_override

    # Kalau auto-detect aktif, coba dua variant arsitektur dan pakai yang cocok
    # dengan checkpoint (lebih reliable daripada menebak dari nama key).
    if auto_detect:
        candidates = [backbone_variant] + [
            v for v in ["retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"] if v != backbone_variant
        ]
    else:
        candidates = [backbone_variant]

    last_error = None
    for variant in candidates:
        try:
            model = build_retinanet(variant, num_classes)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            model.detected_variant = variant
            model.detected_num_classes = num_classes
            return model
        except RuntimeError as e:
            last_error = e
            continue

    raise last_error


def run_retinanet_inference(model, image: Image.Image, conf: float, class_names: list):
    import torchvision.transforms.functional as TF

    img_tensor = TF.to_tensor(image).to(DEVICE)
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    annotated_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_img)
    font = get_scaled_font(image.size)
    box_width = max(2, int(min(image.size) * 0.006))

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score < conf:
            continue

        label = int(label)
        if class_names and 1 <= label <= len(class_names):
            cls_name = class_names[label - 1]
        else:
            cls_name = f"class_{label}"

        color = get_color(label)
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        caption = f"{cls_name} {score:.2f}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        # Beri sedikit padding di sekitar teks supaya tidak terlalu mepet
        padded_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
        draw.rectangle(padded_bbox, fill=color)
        draw.text((x1, y1), caption, fill="black", font=font)

        detections.append({"class": cls_name, "score": round(float(score), 3), "box": [round(v, 1) for v in box]})

    return annotated_img, detections


# --------------------------------------------------------------------------
# Sidebar — konfigurasi model (hanya tampil saat tab 1 aktif)
# --------------------------------------------------------------------------
with tab1:
    st.caption("Pilih model yang mau diuji di sidebar, upload satu gambar, lalu lihat hasil deteksinya.")

with st.sidebar:
    st.header("Konfigurasi Model")

    model_choice = st.selectbox(
        "Model yang digunakan",
        ["YOLOv8s", "RetinaNet", "Kedua Model"],
        index=2,
        help="Pilih model mana yang mau dijalankan. Weight sudah otomatis di-load dari server, "
        "tidak perlu upload manual.",
    )
    use_yolo = model_choice in ("YOLOv8s", "Kedua Model")
    use_retina = model_choice in ("RetinaNet", "Kedua Model")

    st.divider()

    if use_yolo:
        st.subheader("YOLOv8s")
        if os.path.exists(YOLO_MODEL_PATH):
            st.caption(f"✅ Weight ditemukan: `{YOLO_MODEL_PATH}`")
        else:
            st.error(f"❌ Weight tidak ditemukan di `{YOLO_MODEL_PATH}`")
        yolo_conf = st.slider("Confidence threshold (YOLO)", 0.0, 1.0, 0.25, 0.05)
        yolo_iou = st.slider("IoU threshold / NMS (YOLO)", 0.0, 1.0, 0.45, 0.05)
        st.divider()

    if use_retina:
        st.subheader("RetinaNet")
        if os.path.exists(RETINA_MODEL_PATH):
            st.caption(f"✅ Weight ditemukan: `{RETINA_MODEL_PATH}`")
        else:
            st.error(f"❌ Weight tidak ditemukan di `{RETINA_MODEL_PATH}`")
        retina_auto_detect = st.checkbox(
            "Auto-deteksi arsitektur & jumlah class dari checkpoint",
            value=True,
            help="Coba load checkpoint ke retinanet_resnet50_fpn dan retinanet_resnet50_fpn_v2, "
            "lalu pakai yang berhasil cocok. Jumlah class juga dihitung otomatis dari checkpoint.",
        )
        retina_backbone = st.selectbox(
            "Varian arsitektur RetinaNet",
            ["retinanet_resnet50_fpn_v2", "retinanet_resnet50_fpn"],
            help="Pilih sesuai arsitektur yang dipakai saat training.",
        )
        retina_class_names_raw = st.text_area(
            "Nama class",
            value="kardiomegali, normal",
            placeholder="contoh: kucing, anjing, motor",
            help="Urutan harus sama persis dengan id class saat training (index 1, 2, 3, ...).",
        )
        retina_class_names = [c.strip() for c in retina_class_names_raw.split(",") if c.strip()]
        retina_conf = st.slider("Confidence threshold (RetinaNet)", 0.0, 1.0, 0.5, 0.05)
        st.divider()

    st.subheader("Performa")
    resize_enabled = st.checkbox(
        "Perkecil gambar sebelum inferensi (lebih cepat di CPU)", value=True
    )
    max_dim = st.slider(
        "Ukuran sisi terpanjang maksimal (px)",
        min_value=320,
        max_value=2048,
        value=1024,
        step=64,
        disabled=not resize_enabled,
        help="Gambar X-ray resolusi tinggi sangat memperlambat inferensi RetinaNet di CPU. "
        "Perkecil dulu untuk testing yang lebih responsif.",
    )

    st.divider()
    st.caption(f"Device yang dipakai: **{DEVICE}**")


# --------------------------------------------------------------------------
# Tab 1 — Upload gambar & jalankan inferensi langsung
# --------------------------------------------------------------------------
with tab1:
    image_file = st.file_uploader("📤 Upload gambar untuk diuji", type=["jpg", "jpeg", "png", "bmp", "webp"], key="local_image")

    if image_file is not None:
        image = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")

        if resize_enabled:
            image = resize_image_if_needed(image, max_dim)

        run_button = st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True)

        n_result_cols = 1 + int(use_yolo) + int(use_retina)
        cols = st.columns(n_result_cols)
        col_original = cols[0]
        col_idx = 1
        col_yolo = cols[col_idx] if use_yolo else None
        if use_yolo:
            col_idx += 1
        col_retina = cols[col_idx] if use_retina else None

        with col_original:
            st.subheader("Gambar Asli")
            # st.caption(f"Ukuran dipakai: {image.size[0]} x {image.size[1]} px")
            st.image(image, use_container_width=True)

        if run_button:
            # ---------------- YOLOv8s ----------------
            if use_yolo:
                with col_yolo:
                    st.subheader("YOLOv8s")
                    if not os.path.exists(YOLO_MODEL_PATH):
                        st.warning(f"Weight YOLOv8s tidak ditemukan di `{YOLO_MODEL_PATH}`.")
                    else:
                        try:
                            with st.spinner("Menjalankan YOLOv8s..."):
                                yolo_model = load_yolo_model(YOLO_MODEL_PATH)
                                yolo_annotated, yolo_detections = run_yolo_inference(
                                    yolo_model, image, yolo_conf, yolo_iou
                                )
                            st.image(yolo_annotated, use_container_width=True)
                            st.write(f"Jumlah deteksi: **{len(yolo_detections)}**")
                        except Exception as e:
                            st.error(f"Gagal menjalankan YOLOv8s.\n\nDetail error: {e}")

            # ---------------- RetinaNet ----------------
            if use_retina:
                with col_retina:
                    st.subheader("RetinaNet")
                    if not os.path.exists(RETINA_MODEL_PATH):
                        st.warning(f"Weight RetinaNet tidak ditemukan di `{RETINA_MODEL_PATH}`.")
                    else:
                        try:
                            with st.spinner("Menjalankan RetinaNet..."):
                                retina_model = load_retinanet_model(
                                    RETINA_MODEL_PATH,
                                    DEFAULT_RETINA_NUM_CLASSES,
                                    retina_backbone,
                                    retina_auto_detect,
                                )
                                retina_annotated, retina_detections = run_retinanet_inference(
                                    retina_model, image, retina_conf, retina_class_names
                                )
                            # st.caption(
                            #     f"Konfigurasi dipakai: **{retina_model.detected_variant}**, "
                            #     f"**{retina_model.detected_num_classes}** class (termasuk background)"
                            # )
                            st.image(retina_annotated, use_container_width=True)
                            st.write(f"Jumlah deteksi: **{len(retina_detections)}**")
                        except Exception as e:
                            st.error(
                                "Gagal load state_dict ke arsitektur RetinaNet, baik dengan auto-detect maupun manual. "
                                "Kemungkinan checkpoint memakai backbone/anchor custom yang berbeda dari default torchvision.\n\n"
                                f"Detail error: {e}"
                            )
    else:
        st.info("⬆️ Upload gambar terlebih dahulu untuk mulai pengujian.")


# --------------------------------------------------------------------------
# Tab 2 — Test via FastAPI
# --------------------------------------------------------------------------
with tab2:
    st.caption("Kirim gambar ke FastAPI dan tampilkan hasil deteksi yang dikembalikan oleh API.")

    api_url = st.text_input(
        "URL FastAPI",
        value="http://localhost:8000",
        help="Ganti dengan IP server kalau API jalan di komputer lain.",
    )
    api_model = st.selectbox(
        "Model yang dipakai",
        ["both", "yolo", "retinanet"],
        format_func=lambda v: {"both": "Kedua Model", "yolo": "YOLOv8s", "retinanet": "RetinaNet"}[v],
        help="Pilih model mana yang digunakan untuk deteksi.",
    )
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        api_yolo_conf   = st.slider("Confidence YOLO",      0.0, 1.0, 0.25, 0.05, key="api_yolo_conf")
    with col_conf2:
        api_retina_conf = st.slider("Confidence RetinaNet", 0.0, 1.0, 0.50, 0.05, key="api_retina_conf")

    api_image_file = st.file_uploader(
        "📤 Upload gambar untuk dikirim ke API",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="api_image",
    )

    if api_image_file is not None:
        api_image = Image.open(io.BytesIO(api_image_file.getvalue())).convert("RGB")

        api_run = st.button("Kirim ke FastAPI", type="primary", use_container_width=True)

        st.subheader("Gambar Asli")
        st.image(api_image, use_container_width=True)

        if api_run:
            with st.spinner("Mengirim gambar ke FastAPI..."):
                try:
                    response = requests.post(
                        f"{api_url.rstrip('/')}/predict",
                        files={"image": (api_image_file.name, api_image_file.getvalue(), api_image_file.type)},
                        data={
                            "model":       api_model,
                            "yolo_conf":   api_yolo_conf,
                            "retina_conf": api_retina_conf,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.success(f"✅ Response diterima — device API: **{result.get('device', '?')}**, ukuran gambar: **{result.get('image_size', '?')}**")

                    col_yolo_api, col_retina_api = st.columns(2)

                    if "yolo" in result:
                        with col_yolo_api:
                            st.subheader("Hasil YOLOv8s")
                            img_data = base64.b64decode(result["yolo"]["image_base64"])
                            st.image(Image.open(io.BytesIO(img_data)), use_container_width=True)
                            st.write(f"Jumlah deteksi: **{result['yolo']['detection_count']}**")

                    if "retinanet" in result:
                        with col_retina_api:
                            st.subheader("Hasil RetinaNet")
                            img_data = base64.b64decode(result["retinanet"]["image_base64"])
                            st.image(Image.open(io.BytesIO(img_data)), use_container_width=True)
                            st.write(f"Jumlah deteksi: **{result['retinanet']['detection_count']}**")

                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Tidak bisa terhubung ke FastAPI di `{api_url}`. "
                        "Pastikan server FastAPI sudah jalan (`uvicorn api:app --port 8000`)."
                    )
                except requests.exceptions.Timeout:
                    st.error("Request timeout — FastAPI terlalu lama merespons. Coba perkecil gambar atau naikkan timeout.")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("⬆️ Upload gambar terlebih dahulu untuk dikirim ke FastAPI.")