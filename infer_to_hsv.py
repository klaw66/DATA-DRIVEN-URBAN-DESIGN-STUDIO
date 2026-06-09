"""
간판 세그멘테이션 + HSV 분석 통합 batch
================================================
폴더 안의 모든 이미지에 대해:
  1. DeepLabV3+ (Kumakoshi et al. 2021 기반) 모델로 간판 mask 생성
  2. mask 영역의 픽셀들만 HSV 분석 → polar plot + S/V 히스토그램
  3. summary_stats.csv 일괄 저장

전체 facade가 아닌 **간판만** 측색하므로 가로수·차량·하늘이 노이즈로 들어오지 않음.

Requirements:
    pip install numpy matplotlib opencv-python scipy torch torchvision \\
        segmentation-models-pytorch

Pre-trained weights:
    best_model_deeplabv3plus_resnet50.pth (Kumakoshi et al. 학습 가중치)
    → 학습 환경에서 받아서 프로젝트 루트에 두기

Usage:
    1. CONFIG 섹션에서 경로 수정
    2. input_images/ 폴더에 이미지 넣기
    3. python infer_to_hsv.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# =====================================================================
# CONFIG
# =====================================================================
INPUT_DIR        = "input_images"
OUTPUT_DIR       = "output_results"
MODEL_PATH       = "best_model_deeplabv3plus_resnet50.pth"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

# Segmentation 파라미터
THRESHOLD = 0.5       # 마스크 이진화 임계
TILE      = 0         # 가로 타일 너비 (0 = 이미지 높이로 자동)
OVERLAP   = 0.5       # 타일 겹침 비율
ALPHA     = 0.3       # overlay 투명도

# HSV 시각화 파라미터
N_SAMPLE        = 40000
POLAR_MIN_S     = 0.12
HIGH_SAT_THRESH = 0.30
N_HIST_BINS     = 120
RANDOM_SEED     = 42

# =====================================================================
# Segmentation 모델 (infer_billboard.py 기반)
# =====================================================================
IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(device):
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50', encoder_weights=None,
        in_channels=3, classes=1, activation='sigmoid',
    ).to(device)
    model.eval()
    return model


def load_weights(model, path, device):
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, nn.Module):
        return obj.to(device).eval()

    state = obj
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing params: {len(missing)} (e.g. {missing[:2]})")
    if unexpected:
        print(f"  [warn] unexpected params: {len(unexpected)} (e.g. {unexpected[:2]})")
    return model.eval()


def preprocess(rgb):
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))[None]
    return torch.from_numpy(img)


@torch.no_grad()
def infer_prob_512(model, rgb, device):
    x = preprocess(rgb).to(device)
    p = model(x)
    return p.squeeze().float().cpu().numpy()


@torch.no_grad()
def predict_full(model, bgr, device, tile=None, overlap=0.5):
    """원본 BGR → 원본 해상도 확률맵 (긴 가로 이미지는 sliding window)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    if tile is None:
        tile = H
    tile = int(min(tile, W))

    if W <= tile * 1.2:
        p = infer_prob_512(model, rgb, device)
        return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)

    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, stride))
    if xs[-1] != W - tile:
        xs.append(W - tile)

    prob = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    wx = 1.0 - np.abs(np.linspace(-1, 1, tile))
    wx = np.clip(wx, 0.05, 1.0)
    wmask = np.broadcast_to(wx, (H, tile)).astype(np.float32)

    for x in xs:
        crop = rgb[:, x:x + tile]
        p = infer_prob_512(model, crop, device)
        p = cv2.resize(p, (tile, H), interpolation=cv2.INTER_LINEAR)
        prob[:, x:x + tile] += p * wmask
        wsum[:, x:x + tile] += wmask

    prob /= np.maximum(wsum, 1e-6)
    return prob


def make_overlay(bgr, mask, alpha=0.3):
    overlay = bgr.copy()
    red = np.zeros_like(bgr)
    red[..., 2] = 255
    m = mask > 0
    overlay[m] = (alpha * red[m] + (1 - alpha) * bgr[m]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    return overlay


# =====================================================================
# HSV 시각화 (마스킹된 픽셀만)
# =====================================================================
def hsv_visualization_masked(bgr, sign_mask, out_path, title):
    """sign_mask=True인 픽셀들만 HSV 분석 + 시각화."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.float32) / 179.0
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0

    # mask 영역만 추출
    H_m = H[sign_mask]
    S_m = S[sign_mask]
    V_m = V[sign_mask]

    n_sign = int(sign_mask.sum())
    if n_sign == 0:
        print(f"  [skip] no signage detected")
        return None

    stats = {
        'sign_pixels': n_sign,
        'sign_area_pct': sign_mask.mean() * 100,
        'Mean H (deg)': H_m.mean() * 360,
        'Mean S':       S_m.mean(),
        'Std S':        S_m.std(),
        'Mean V':       V_m.mean(),
        'Std V':        V_m.std(),
        'High-chroma (S>0.5) %': (S_m > 0.5).mean() * 100,
        'Low-chroma  (S<0.1) %': (S_m < 0.1).mean() * 100,
    }

    # Sample for polar
    np.random.seed(RANDOM_SEED)
    n_sample = min(N_SAMPLE, n_sign)
    idx = np.random.choice(n_sign, n_sample, replace=False)
    H_s, S_s, V_s = H_m[idx], S_m[idx], V_m[idx]

    mask = S_s > POLAR_MIN_S
    H_polar = H_s[mask]
    S_polar = S_s[mask]
    V_polar = V_s[mask]

    H_jit = H_polar + np.random.uniform(-0.008, 0.008, H_polar.shape)
    S_jit = np.clip(S_polar + np.random.uniform(-0.02, 0.02, S_polar.shape), 0, 1)
    rgb_pts = hsv_to_rgb(np.stack([H_polar, S_polar, V_polar], axis=1))

    # Figure
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(15, 7.5), facecolor='#f0f0f0')
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
        wspace=0.10, hspace=0.32,
        left=0.04, right=0.97, top=0.92, bottom=0.08,
    )

    ax_h = fig.add_subplot(gs[:, 0], projection='polar')
    ax_h.set_facecolor('#f0f0f0')

    theta = H_jit * 2 * np.pi
    radius = S_jit
    high_mask = S_polar > HIGH_SAT_THRESH
    low_mask = ~high_mask

    ax_h.scatter(theta[low_mask], radius[low_mask],
                 c=rgb_pts[low_mask], s=3.5, alpha=0.40,
                 edgecolors='none', rasterized=True)
    ax_h.scatter(theta[high_mask], radius[high_mask],
                 c=rgb_pts[high_mask], s=8.0, alpha=0.85,
                 edgecolors='none', rasterized=True)

    ax_h.set_ylim(0, 1)
    ax_h.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_h.set_yticklabels([])
    ax_h.set_rlabel_position(80)
    munsell = ['5R', '5YR', '5Y', '5GY', '5G', '5BG', '5B', '5PB', '5P', '5RP']
    angles_main = np.linspace(0, 2 * np.pi, 11)[:-1]
    ax_h.set_xticks(angles_main)
    ax_h.set_xticklabels([])
    for ang, lab in zip(angles_main, munsell):
        ax_h.text(ang, 1.13, lab, ha='center', va='center',
                  fontsize=11, fontweight='bold', color='#333')
    for r in [0.2, 0.4, 0.6, 0.8]:
        ax_h.text(np.deg2rad(80), r, f'{r:.1f}',
                  fontsize=7, color='#aaa', ha='center', va='bottom')
    ax_h.grid(True, alpha=0.25, linewidth=0.4, color='#999')
    ax_h.spines['polar'].set_color('#bbb')
    ax_h.text(-0.02, 1.02, 'H', transform=ax_h.transAxes,
              fontsize=28, fontweight='bold', color='#222')
    ax_h.set_title(f"{title}  (signage area: {stats['sign_area_pct']:.1f}%)",
                   fontsize=11, pad=30, color='#555')

    def gradient_hist(ax, data, n_bins, color_func, label_letter):
        counts, edges = np.histogram(data, bins=n_bins, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2
        width = 1 / n_bins
        bar_colors = np.array([color_func(c) for c in centers])
        ax.bar(centers, counts, width=width * 1.02, color=bar_colors,
               edgecolor='none', align='center', linewidth=0)
        ax.set_facecolor('#b8b8b8')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([])
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#888')
        ax.tick_params(axis='x', colors='#333', labelsize=9)
        ax.text(0.015, 0.88, label_letter, transform=ax.transAxes,
                fontsize=24, fontweight='bold', color='#1a1a1a')

    def s_color(s_val):
        return (1.0, 1.0 - 0.85 * s_val, 1.0 - 0.85 * s_val)

    def v_color(v_val):
        return (v_val, v_val, min(1.0, v_val * 1.05 + 0.02))

    ax_s = fig.add_subplot(gs[0, 1])
    gradient_hist(ax_s, S_m, N_HIST_BINS, s_color, 'S')

    ax_v = fig.add_subplot(gs[1, 1])
    gradient_hist(ax_v, V_m, N_HIST_BINS, v_color, 'V')

    plt.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='#f0f0f0')
    plt.close(fig)
    return stats


# =====================================================================
# Main
# =====================================================================
def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    if not in_dir.exists():
        print(f"❌ 입력 폴더 없음: {in_dir.resolve()}")
        return
    if not Path(MODEL_PATH).exists():
        print(f"❌ 모델 가중치 없음: {MODEL_PATH}")
        print(f"   학습된 .pth 파일을 프로젝트 루트에 두세요.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not img_paths:
        print(f"❌ {in_dir} 안에 이미지 없음")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {device}")
    print(f"✓ Found {len(img_paths)} images in {in_dir}")
    print(f"  → output: {out_dir.resolve()}\n")

    # Load model once
    print(f"Loading model: {MODEL_PATH}")
    model = build_model(device)
    model = load_weights(model, MODEL_PATH, device)
    print("Model ready.\n")

    # Process each image
    all_stats = []
    for i, img_path in enumerate(img_paths, 1):
        print(f"[{i}/{len(img_paths)}] {img_path.name}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"   ❌ 읽기 실패")
            continue

        # 1) Segmentation
        tile = TILE if TILE > 0 else None
        prob = predict_full(model, bgr, device, tile=tile, overlap=OVERLAP)
        sign_mask = prob > THRESHOLD

        # 2) Save mask / overlay (디버깅용)
        base = img_path.stem
        cv2.imwrite(str(out_dir / f"{base}_mask.png"),
                    sign_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / f"{base}_overlay.jpg"),
                    make_overlay(bgr, sign_mask.astype(np.uint8) * 255, alpha=ALPHA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 3) HSV analysis (masked)
        hsv_out = out_dir / f"{base}_hsv.png"
        stats = hsv_visualization_masked(bgr, sign_mask, hsv_out, base)
        if stats is None:
            continue

        stats = {'file': img_path.name, **stats}
        all_stats.append(stats)
        print(f"   sign area: {stats['sign_area_pct']:.1f}%  "
              f"Mean S: {stats['Mean S']:.3f}  Mean V: {stats['Mean V']:.3f}  "
              f"High-chroma: {stats['High-chroma (S>0.5) %']:.1f}%")

    # Summary CSV
    if all_stats:
        csv_path = out_dir / "summary_stats.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n✓ Done. {len(all_stats)} images processed.")
        print(f"  → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
