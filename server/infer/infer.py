import argparse
import json
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import TwoStageModel

# === Occlusion sensitivity
def occlusion_sensitivity_glaucoma(two_stage_model,
                                   image,          # 1×3×H×W
                                   patch_size=32, stride=16,
                                   device='cuda'):
    two_stage_model.eval()
    image = image.to(device)
    C, H, W = image.shape[1:]

    heatmap = np.zeros(((H - patch_size)//stride + 1, (W - patch_size)//stride + 1))

    with torch.no_grad():
        _, g_logit = two_stage_model(image)
        base_prob = torch.sigmoid(g_logit)[0].item()

    idx_h = 0
    for y in range(0, H - patch_size + 1, stride):
        idx_w = 0
        for x in range(0, W - patch_size + 1, stride):
            occ_img = image.clone()
            occ_img[:, :, y:y+patch_size, x:x+patch_size] = 0

            with torch.no_grad():
                _, g_logit_occ = two_stage_model(occ_img)
                prob_occ = torch.sigmoid(g_logit_occ)[0].item()

            heatmap[idx_h, idx_w] = base_prob - prob_occ  # drop-in-prob
            idx_w += 1
        idx_h += 1

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap

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

    occ_map = occlusion_sensitivity_glaucoma(model, inp, patch_size=32, stride=16, device=device)
    save_occlusion_overlay(img_pil, occ_map, os.path.join(output_dir, f"combined.png"))

    # # Top-4 sign 선택
    # top4_idx = np.argsort(sign_probs)[-4:][::-1]

    # # 개별 heatmap 저장
    # occlusion_maps = []
    # for i, idx in enumerate(top4_idx):
    #     occ_map = occlusion_sensitivity(model.sign_classifier, inp, idx, device=device)
    #     occlusion_maps.append(occ_map)
    #     save_occlusion_overlay(img_pil, occ_map, os.path.join(output_dir, f"{Signs[i]}.png"))

    # # Combined heatmap 저장
    # combined = np.mean(occlusion_maps, axis=0)
    # save_occlusion_overlay(img_pil, combined, os.path.join(output_dir, f"combined.png"))

    print(f"Inference completed. Results saved to {output_dir}")

# === CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str, default="./best_glaucoma_classifier.pt")
    args = parser.parse_args()

    run_inference(args.image_path, args.output_dir, args.model_path)
