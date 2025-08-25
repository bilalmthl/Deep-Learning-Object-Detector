import argparse
from flask import Flask, request, jsonify
from PIL import Image
import torch
from utils import read_classes, load_model, make_transform, build_model

app = Flask(__name__)

CLASSES = None
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TFM = make_transform()

@torch.inference_mode()
def predict_pil(img):
    x = TFM(img).unsqueeze(0)
    logits = MODEL(x.to(DEVICE))
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = probs.argmax()
    return CLASSES[idx], probs

@app.route("/ping")
def ping():
    return jsonify(ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="missing file field 'image'"), 400
    img = Image.open(request.files["image"].stream).convert("RGB")
    label, probs = predict_pil(img)
    display = {"apples":"APPLE", "bananas":"BANANA", "oranges":"ORANGE", "none":"NONE"}.get(label, label.upper())
    return jsonify(
        label=label,
        display=display,
        probs={c: float(probs[i]) for i, c in enumerate(CLASSES)}
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/best_model.pth")
    parser.add_argument("--classes", default="artifacts/classes.txt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global CLASSES, MODEL
    CLASSES = read_classes(args.classes)
    MODEL = load_model(args.model, len(CLASSES), DEVICE)

    print("Device:", DEVICE)
    print("Classes:", CLASSES)
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
