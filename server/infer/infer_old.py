import argparse
import json
import os
import torch
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import TwoStageModel

# === Occlusion sensitivity
def occlusion_sensitivity(model, image, target_class, patch_size=32, stride=16, device='cuda'):
    model.eval()
    image = image.to(device)
    C, H, W = image.shape[1:]

    heatmap = np.zeros(((H - patch_size)//stride + 1, (W - patch_size)//stride + 1))
    idx_h = 0
    for y in range(0, H - patch_size + 1, stride):
        idx_w = 0
        for x in range(0, W - patch_size + 1, stride):
            occluded = image.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0

            with torch.no_grad():
                logits = model(occluded)
                probs = torch.sigmoid(logits)
                score = probs[0, target_class].item()

            heatmap[idx_h, idx_w] = score
            idx_w += 1
        idx_h += 1

    base_pred = torch.sigmoid(model(image))[0, target_class].item()
    sensitivity_map = base_pred - heatmap
    sensitivity_map = np.maximum(sensitivity_map, 0)
    sensitivity_map /= (sensitivity_map.max() + 1e-8)
    return sensitivity_map

# === Overlay image
def save_occlusion_overlay(image_pil, occlusion_map, save_path):
    img_np = np.array(image_pil).astype(np.float32) / 255.0
    occlusion_map_resized = cv2.resize(occlusion_map, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * (1.0 - occlusion_map_resized)), cv2.COLORMAP_JET).astype(np.float32) / 255.0
    overlay = heatmap * 0.4 + img_np * 0.6
    overlay /= overlay.max()

    plt.imsave(save_path, overlay)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# === 메인 추론 함수
def run_inference(image_path, output_dir, model_path):
    # Delete any old results, then recreate folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TwoStageModel(num_signs=10, pretrained=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    # Transform (반드시 학습시 사용한 transform과 동일)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 이미지 로드
    img = cv2.imread(image_path)
    img = apply_clahe(img)
    
    # Save CLAHE result
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, "preprocessed_" + filename) 
    cv2.imwrite(save_path, img)

    img_pil = Image.fromarray(img)
    inp = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        sign_logits, glaucoma_logits = model(inp)
        sign_probs = torch.sigmoid(sign_logits).cpu().numpy().flatten()
        glaucoma_prob = torch.sigmoid(glaucoma_logits).cpu().item()

    # JSON result
    result = {
        "prediction": "glaucoma" if glaucoma_prob >= 0.65 else "normal",
        "prediction_score": round(float(glaucoma_prob),4)
    }

    Signs =  ['ANRS', 'DH', 'RNFLDS', 'ANRI', 'BCLVI', 'NVT', 'BCLVS', 'LD', 'RNFLDI', 'LC' ]
    for i, prob in enumerate(sign_probs):
        sign_name = Signs[i]
        result[sign_name] = round(float(prob), 4)

    json_path = os.path.join(output_dir, "result.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)

    # Top-4 sign indices, descending by probability
    top4_idx = np.argsort(sign_probs)[-4:][::-1]

    print(top4_idx)
    occlusion_maps = []
    for idx in top4_idx:
        occ_map = occlusion_sensitivity(model.sign_classifier, inp, idx, device=device)
        occlusion_maps.append(occ_map)

        # Use the *actual* sign name at index `idx`
        sign_name = Signs[idx]
        save_occlusion_overlay(
            img_pil,
            occ_map,
            os.path.join(output_dir, f"{sign_name}.png")
        )

    # Combined heatmap 저장
    combined = np.mean(occlusion_maps, axis=0)
    save_occlusion_overlay(img_pil, combined, os.path.join(output_dir, f"combined.png"))

    print(f"Inference completed. Results saved to {output_dir}")

# === CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str, default="./best_glaucoma_classifier.pt")
    args = parser.parse_args()

    run_inference(args.image_path, args.output_dir, args.model_path)
