import argparse
from PIL import Image
import torch
from utils import read_classes, load_model, make_transform, build_model

def predict_image(img_path, model_path, classes_path, device):
    classes = read_classes(classes_path)
    model = load_model(model_path, len(classes), device)
    tfm = make_transform()

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
        pred_idx = probs.argmax()
    
    return classes[pred_idx], probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="artifacts/best_model.pth")
    parser.add_argument("--classes", default="artifacts/classes.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label, probs = predict_image(args.image, args.model, args.classes, device)
    print("Prediction:", label)
    print("Probabilities:", dict(zip(read_classes(args.classes), probs)))

if __name__ == "__main__":
    main()
