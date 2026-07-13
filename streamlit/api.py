
import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Konfigurasi — sesuaikan path weight model di sini
# --------------------------------------------------------------------------
YOLO_WEIGHT_PATH  = os.getenv("YOLO_WEIGHT",  "../runs/yolov8s_kardio/weights/best.pt")
RETINA_WEIGHT_PATH = os.getenv("RETINA_WEIGHT", "../outputs/retinanet_kardio_best.pt")
RETINA_CLASS_NAMES = [
    c.strip() for c in os.getenv("RETINA_CLASSES", "kardiomegali,normal").split(",")
]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "800"))

# --------------------------------------------------------------------------
# State global model (di-load sekali saat startup)
# --------------------------------------------------------------------------
models: dict = {}


# --------------------------------------------------------------------------
# Helper umum
# --------------------------------------------------------------------------
def resize_if_needed(image: Image.Image, max_dim: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def get_color(label: int):
    rng = np.random.default_rng(label * 9973 + 17)
    return tuple(int(x) for x in rng.integers(50, 230, size=3))


def get_scaled_font(image_size: tuple, scale: float = 0.03):
    size = max(16, int(min(image_size) * scale))
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------------------------------------------------------
# Loader model
# --------------------------------------------------------------------------
def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint and not any(
        k.startswith(("backbone", "head")) for k in checkpoint.keys()
    ):
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _infer_num_classes(state_dict, num_anchors: int = 9) -> int:
    key = "head.classification_head.cls_logits.weight"
    if key in state_dict:
        out = state_dict[key].shape[0]
        if out % num_anchors == 0:
            return out // num_anchors
    return 91


def load_yolo_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def load_retinanet_model(path: str):
    checkpoint  = torch.load(path, map_location=DEVICE)
    state_dict  = _extract_state_dict(checkpoint)
    num_classes = _infer_num_classes(state_dict)

    for variant in ["retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"]:
        try:
            if variant == "retinanet_resnet50_fpn_v2":
                model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                    weights=None, weights_backbone=None, num_classes=num_classes
                )
            else:
                model = torchvision.models.detection.retinanet_resnet50_fpn(
                    weights=None, weights_backbone=None, num_classes=num_classes
                )
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            return model
        except RuntimeError:
            continue

    raise RuntimeError("Gagal load RetinaNet — cek path dan arsitektur checkpoint.")


# --------------------------------------------------------------------------
# Inferensi
# --------------------------------------------------------------------------
def infer_yolo(model, image: Image.Image, conf: float, iou: float):
    results = model.predict(source=np.array(image), conf=conf, iou=iou, verbose=False)
    r = results[0]
    annotated = Image.fromarray(r.plot()[:, :, ::-1])
    return annotated, len(r.boxes)


def infer_retinanet(model, image: Image.Image, conf: float, class_names: list):
    img_tensor = TF.to_tensor(image).to(DEVICE)
    with torch.no_grad():
        preds = model([img_tensor])[0]

    boxes  = preds["boxes"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    out   = image.copy().convert("RGB")
    draw  = ImageDraw.Draw(out)
    font  = get_scaled_font(image.size)
    bw    = max(2, int(min(image.size) * 0.006))
    count = 0

    for box, label, score in zip(boxes, labels, scores):
        if score < conf:
            continue
        count += 1
        label    = int(label)
        cls_name = (class_names[label - 1]
                    if class_names and 1 <= label <= len(class_names)
                    else f"class_{label}")
        color = get_color(label)
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=bw)
        caption  = f"{cls_name} {score:.2f}"
        tb       = draw.textbbox((x1, y1), caption, font=font)
        padded   = (tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2)
        draw.rectangle(padded, fill=color)
        draw.text((x1, y1), caption, fill="black", font=font)

    return out, count


# --------------------------------------------------------------------------
# Lifespan — load model sekali saat server start
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(YOLO_WEIGHT_PATH):
        try:
            models["yolo"] = load_yolo_model(YOLO_WEIGHT_PATH)
            print(f"✅ YOLOv8s loaded: {YOLO_WEIGHT_PATH}")
        except Exception as e:
            print(f"⚠️  Gagal load YOLO: {e}")
    else:
        print(f"⚠️  YOLO weight tidak ditemukan: {YOLO_WEIGHT_PATH}")

    if os.path.exists(RETINA_WEIGHT_PATH):
        try:
            models["retina"] = load_retinanet_model(RETINA_WEIGHT_PATH)
            print(f"✅ RetinaNet loaded: {RETINA_WEIGHT_PATH}")
        except Exception as e:
            print(f"⚠️  Gagal load RetinaNet: {e}")
    else:
        print(f"⚠️  RetinaNet weight tidak ditemukan: {RETINA_WEIGHT_PATH}")

    yield
    models.clear()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(
    title="X-Ray Kardiomegali Detection API",
    description=(
        "API inferensi YOLOv8s dan RetinaNet untuk deteksi kardiomegali pada X-ray thorax. "
        "Kirim gambar lewat POST /predict, dapatkan gambar anotasi dalam format base64."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — izinkan frontend dari domain manapun (ubah allow_origins di production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/")
def root():
    """Info server dan model yang sedang aktif."""
    return {
        "status": "running",
        "models_loaded": list(models.keys()),
        "device": DEVICE,
        "retina_classes": RETINA_CLASS_NAMES,
    }


@app.get("/health")
def health():
    """Health check — untuk monitoring atau load balancer."""
    return {"status": "ok", "models": list(models.keys())}


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Gambar X-ray (JPG/PNG/BMP/WEBP)"),
    model: str = Form(
        "both",
        description="Model yang dipakai: 'yolo', 'retinanet', atau 'both'",
    ),
    yolo_conf:   float = Form(0.25, description="Confidence threshold YOLOv8s"),
    yolo_iou:    float = Form(0.45, description="IoU / NMS threshold YOLOv8s"),
    retina_conf: float = Form(0.50, description="Confidence threshold RetinaNet"),
):
    """
    Jalankan deteksi pada gambar X-ray.

    **Response JSON:**
    ```json
    {
      "filename": "foto.jpg",
      "image_size": [800, 800],
      "device": "cpu",
      "yolo": {
        "image_base64": "<string PNG base64>",
        "detection_count": 1
      },
      "retinanet": {
        "image_base64": "<string PNG base64>",
        "detection_count": 1
      }
    }
    ```

    Frontend bisa tampilkan gambar dengan:
    `<img src="data:image/png;base64,{image_base64}" />`
    """
    SUPPORTED = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if image.content_type not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: {image.content_type}. Gunakan JPG, PNG, BMP, atau WEBP.",
        )

    if model not in ("yolo", "retinanet", "both"):
        raise HTTPException(status_code=400, detail="Parameter 'model' harus 'yolo', 'retinanet', atau 'both'.")

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = resize_if_needed(img, MAX_IMAGE_DIM)

    response = {
        "filename": image.filename,
        "image_size": list(img.size),
        "device": DEVICE,
    }

    if model in ("yolo", "both"):
        if "yolo" not in models:
            raise HTTPException(status_code=503, detail="Model YOLO belum ter-load. Cek path weight di server.")
        annotated, count = infer_yolo(models["yolo"], img, yolo_conf, yolo_iou)
        response["yolo"] = {
            "image_base64": pil_to_base64(annotated),
            "detection_count": count,
        }

    if model in ("retinanet", "both"):
        if "retina" not in models:
            raise HTTPException(status_code=503, detail="Model RetinaNet belum ter-load. Cek path weight di server.")
        annotated, count = infer_retinanet(models["retina"], img, retina_conf, RETINA_CLASS_NAMES)
        response["retinanet"] = {
            "image_base64": pil_to_base64(annotated),
            "detection_count": count,
        }

    return response