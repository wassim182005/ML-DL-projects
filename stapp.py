import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image


st.set_page_config(page_title="Intel Scene Classifier", page_icon="🖼️", layout="centered")

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "model_artifacts"
STATE_DICT_PATH = ARTIFACTS_DIR / "intel_resnet50_state_dict.pth"
METADATA_PATH = ARTIFACTS_DIR / "intel_resnet50_metadata.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model


@st.cache_resource
def load_model_and_classes() -> tuple[nn.Module, list[str]]:
    if not METADATA_PATH.exists() or not STATE_DICT_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts missing. Expected files:\n"
            f"- {METADATA_PATH}\n"
            f"- {STATE_DICT_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    classes = metadata["classes"]
    num_classes = metadata["num_classes"]

    model = build_model(num_classes)
    state_dict = torch.load(STATE_DICT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, classes


st.title("Intel Scene Classification")
st.caption(f"Running on: {DEVICE}")

try:
    model, classes = load_model_and_classes()
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    st.subheader(f"Prediction: {classes[pred_idx]}")
    st.write(f"Confidence: {probs[pred_idx].item() * 100:.2f}%")

    top_k = min(6, len(classes))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    st.markdown("Top predictions")
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        st.write(f"{rank}. {classes[idx]}: {prob * 100:.2f}%")