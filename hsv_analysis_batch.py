"""
명동 facade HSV batch 분석 — 폴더 안의 모든 이미지 일괄 처리

Requirements:
    pip install numpy matplotlib opencv-python

Usage:
    1. 아래 CONFIG에서 INPUT_DIR, OUTPUT_DIR 수정
    2. INPUT_DIR 폴더에 이미지 파일들 넣기 (.png .jpg .jpeg .bmp .tiff .webp)
    3. python hsv_analysis_batch.py
    4. OUTPUT_DIR에 각 이미지별 *_hsv.png 결과 + summary_stats.csv 생성됨
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2

# =====================================================================
# CONFIG — 여기만 수정
# =====================================================================
INPUT_DIR        = "input_images"        # 입력 이미지 폴더
OUTPUT_DIR       = "output_results"      # 결과 저장 폴더
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

# 분석 파라미터
N_SAMPLE        = 40000
POLAR_MIN_S     = 0.12
HIGH_SAT_THRESH = 0.30
N_HIST_BINS     = 120
RANDOM_SEED     = 42

# =====================================================================
# 분석 함수
# =====================================================================
def analyze_image(img_path, out_path):
    """단일 이미지 → HSV polar + S/V histogram 저장.
    Returns: stats dict
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"읽기 실패: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    H = img_hsv[..., 0].astype(np.float32) / 179.0
    S = img_hsv[..., 1].astype(np.float32) / 255.0
    V = img_hsv[..., 2].astype(np.float32) / 255.0

    stats = {
        'Mean H (deg)': H.mean() * 360,
        'Mean S':       S.mean(),
        'Std S':        S.std(),
        'Mean V':       V.mean(),
        'Std V':        V.std(),
        'High-chroma (S>0.5) %': (S > 0.5).mean() * 100,
        'Low-chroma  (S<0.1) %': (S < 0.1).mean() * 100,
    }

    # Sample for polar
    np.random.seed(RANDOM_SEED)
    n_sample = min(N_SAMPLE, H.size)
    idx = np.random.choice(H.size, n_sample, replace=False)
    H_s, S_s, V_s = H.flatten()[idx], S.flatten()[idx], V.flatten()[idx]

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
        left=0.04, right=0.97, top=0.93, bottom=0.08,
    )

    # H polar
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
    ax_h.spines['polar'].set_linewidth(0.8)
    ax_h.text(-0.02, 1.02, 'H', transform=ax_h.transAxes,
              fontsize=28, fontweight='bold', color='#222')

    # Filename as title
    ax_h.set_title(Path(img_path).name, fontsize=11, pad=30, color='#555')

    # Histogram helper
    def gradient_hist(ax, data, n_bins, color_func, label_letter):
        counts, edges = np.histogram(data.flatten(), bins=n_bins, range=(0, 1))
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
    gradient_hist(ax_s, S, N_HIST_BINS, s_color, 'S')

    ax_v = fig.add_subplot(gs[1, 1])
    gradient_hist(ax_v, V, N_HIST_BINS, v_color, 'V')

    plt.savefig(str(out_path), dpi=180, bbox_inches='tight', facecolor='#f0f0f0')
    plt.close(fig)

    return stats

# =====================================================================
# Main — 폴더 순회
# =====================================================================
if __name__ == "__main__":
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    if not in_dir.exists():
        print(f"❌ 입력 폴더 없음: {in_dir.resolve()}")
        print(f"   이 폴더를 만들고 이미지 파일을 넣어주세요.")
        exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 검색
    img_paths = sorted([
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not img_paths:
        print(f"❌ {in_dir} 안에 이미지 없음")
        print(f"   지원 확장자: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        exit(1)

    print(f"✓ Found {len(img_paths)} images in {in_dir}")
    print(f"  → output: {out_dir.resolve()}\n")

    # 일괄 처리
    all_stats = []
    for i, img_path in enumerate(img_paths, 1):
        out_name = img_path.stem + "_hsv.png"
        out_path = out_dir / out_name
        print(f"[{i}/{len(img_paths)}] {img_path.name}")

        try:
            stats = analyze_image(img_path, out_path)
            stats = {'file': img_path.name, **stats}
            all_stats.append(stats)
            print(f"   Mean S: {stats['Mean S']:.3f}  "
                  f"Mean V: {stats['Mean V']:.3f}  "
                  f"High-chroma: {stats['High-chroma (S>0.5) %']:.1f}%")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    # 요약 CSV 저장
    if all_stats:
        csv_path = out_dir / "summary_stats.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n✓ Saved {len(all_stats)} result images + summary_stats.csv")
        print(f"  → {out_dir.resolve()}")
