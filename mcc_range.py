"""
Myeongdong Color Code (MCC) — 범위 기반 주색 정의 + 적합성 검증 함수

명동 facade의 hue 분포에서 자동으로 두 peak를 찾아 주색 범위로 정의.
S(채도)·V(명도)는 명동 데이터의 하한(percentile)만 적용 — 고채도·고명도는 허용.

Requirements:
    pip install numpy matplotlib opencv-python scipy

Usage:
    1. 상단 CONFIG에서 IMG_PATH, OUT_PATH 수정
    2. python mcc_range.py
    3. 다른 코드에서 import해서 함수 재사용 가능:
       from mcc_range import check_mcc, check_mcc_image
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import cv2
from scipy.signal import find_peaks

# =====================================================================
# CONFIG — 여기만 수정
# =====================================================================
IMG_PATH = "myeongdong_strip_4.png"
OUT_PATH = "myeongdong_mcc_ranges.png"

# 회색·그림자 제외
S_MIN = 0.25
V_MIN = 0.20

# 주색 범위 도출 파라미터
HUE_BIN_DEG     = 5       # hue histogram bin 크기
HUE_RANGE_WIDTH = 50      # peak 주변 ± 도수 (총 폭 = 2*WIDTH)
N_PEAKS         = 2       # 주색 개수

# S, V 하한 (percentile). 상한은 1.0 고정 — 명동 정체성은 고채도 허용
S_PCT_LOW = 5
V_PCT_LOW = 10

# MCC 적합 판정 임계
COMPLIANCE_THRESHOLD = 70.0   # 간판의 chromatic 픽셀 70% 이상이 MCC 범위 안이면 적합

# =====================================================================
# 1. 이미지 분석 (스크립트 단독 실행 시)
# =====================================================================
def compute_mcc_spec(img_path):
    """이미지에서 MCC 범위 자동 도출"""
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지 못 찾음: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.float32) / 179.0
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0

    # Chromatic pixel만
    m = (S > S_MIN) & (V > V_MIN)
    H_c, S_c, V_c = H[m], S[m], V[m]

    # Hue peak 찾기 (circular)
    H_deg = H_c * 360
    n_bins = int(360 / HUE_BIN_DEG)
    hist, edges = np.histogram(H_deg, bins=n_bins, range=(0, 360))
    hist_ext = np.concatenate([hist, hist, hist])
    peaks_ext, _ = find_peaks(hist_ext, distance=int(60 / HUE_BIN_DEG))
    peaks = [p - n_bins for p in peaks_ext if n_bins <= p < 2*n_bins]
    heights = [hist[p] for p in peaks]
    top = sorted(zip(heights, peaks), reverse=True)[:N_PEAKS]
    peak_centers = [edges[p] + HUE_BIN_DEG/2 for _, p in top]

    s_lo = float(np.percentile(S_c, S_PCT_LOW))
    v_lo = float(np.percentile(V_c, V_PCT_LOW))

    regions = []
    for i, pc in enumerate(peak_centers):
        hs = (pc - HUE_RANGE_WIDTH) % 360
        he = (pc + HUE_RANGE_WIDTH) % 360
        regions.append({
            'name': f'MCC-{i+1}',
            'hue_center': pc,
            'hue_range': (hs, he),
            's_range': (s_lo, 1.0),
            'v_range': (v_lo, 1.0),
        })

    return {
        'regions': regions,
        'image_data': {'H': H, 'S': S, 'V': V, 'mask': m,
                       'H_c': H_c, 'S_c': S_c, 'V_c': V_c,
                       'hist': hist, 'edges': edges},
    }

# =====================================================================
# 2. 검증 함수 (import해서 다른 코드에서 재사용 가능)
# =====================================================================
def is_in_hue_range(h_deg, start, end):
    """Circular hue range 검사 (wraparound 처리)"""
    if start <= end:
        return start <= h_deg <= end
    return h_deg >= start or h_deg <= end

def check_mcc(h_norm, s, v, regions):
    """한 픽셀의 MCC 적합 여부 검증.
    Returns: (bool, region_name or None)
    """
    h_deg = h_norm * 360
    for r in regions:
        if (is_in_hue_range(h_deg, *r['hue_range']) and
            r['s_range'][0] <= s <= r['s_range'][1] and
            r['v_range'][0] <= v <= r['v_range'][1]):
            return True, r['name']
    return False, None

def check_mcc_image(img_path, regions,
                    s_min=S_MIN, v_min=V_MIN,
                    compliance_threshold=COMPLIANCE_THRESHOLD):
    """간판 이미지 한 장의 MCC 적합도 판정.
    Returns: dict — compliance_pct, compliant, region_share 등
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지 못 찾음: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0].astype(np.float32) / 179.0
    S = hsv[..., 1].astype(np.float32) / 255.0
    V = hsv[..., 2].astype(np.float32) / 255.0

    chroma = (S > s_min) & (V > v_min)
    n_chroma = int(chroma.sum())
    if n_chroma == 0:
        return {'compliant': False, 'compliance_pct': 0.0,
                'reason': 'no chromatic pixels'}

    h_flat = H[chroma]
    s_flat = S[chroma]
    v_flat = V[chroma]

    n_in = 0
    counts = {r['name']: 0 for r in regions}
    for hi, si, vi in zip(h_flat, s_flat, v_flat):
        ok, name = check_mcc(hi, si, vi, regions)
        if ok:
            n_in += 1
            counts[name] += 1

    pct = n_in / n_chroma * 100
    return {
        'compliance_pct': pct,
        'compliant': pct >= compliance_threshold,
        'chromatic_pixels': n_chroma,
        'in_mcc_pixels': n_in,
        'region_share': {k: v/n_chroma*100 for k, v in counts.items()},
    }

# =====================================================================
# 3. 시각화
# =====================================================================
def to_munsell(h_deg):
    labels_m = ['5R', '5YR', '5Y', '5GY', '5G',
                '5BG', '5B', '5PB', '5P', '5RP']
    return labels_m[int(round(h_deg / 36)) % 10]

def visualize(spec, result, out_path):
    regions = spec['regions']
    d = spec['image_data']
    H_c, S_c, V_c = d['H_c'], d['S_c'], d['V_c']
    hist, edges = d['hist'], d['edges']

    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig = plt.figure(figsize=(15, 7), facecolor='#f0f0f0')
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.4, 1], height_ratios=[1.2, 1],
        wspace=0.18, hspace=0.30,
        left=0.04, right=0.96, top=0.93, bottom=0.06,
    )

    # Polar plot
    ax_p = fig.add_subplot(gs[:, 0], projection='polar')
    ax_p.set_facecolor('#f0f0f0')

    np.random.seed(42)
    n_sample = min(25000, len(H_c))
    si = np.random.choice(len(H_c), n_sample, replace=False)
    H_s, S_s, V_s = H_c[si], S_c[si], V_c[si]
    rgb_pts = hsv_to_rgb(np.stack([H_s, S_s, V_s], axis=1))

    in_mcc = np.array([check_mcc(h, s, v, regions)[0]
                       for h, s, v in zip(H_s, S_s, V_s)])

    ax_p.scatter(H_s[~in_mcc] * 2*np.pi, S_s[~in_mcc],
                 c=rgb_pts[~in_mcc], s=3, alpha=0.20,
                 edgecolors='none', rasterized=True)
    ax_p.scatter(H_s[in_mcc] * 2*np.pi, S_s[in_mcc],
                 c=rgb_pts[in_mcc], s=6, alpha=0.80,
                 edgecolors='none', rasterized=True)

    sector_colors = ['#ff6b35', '#3a7ca5']
    for i, r in enumerate(regions):
        hs, he = r['hue_range']
        if hs > he:
            thetas = np.concatenate([
                np.linspace(np.deg2rad(hs), np.deg2rad(360), 50),
                np.linspace(0, np.deg2rad(he), 50)])
        else:
            thetas = np.linspace(np.deg2rad(hs), np.deg2rad(he), 100)
        r_in, r_out = r['s_range'][0], r['s_range'][1]
        ax_p.fill_between(thetas, r_in, r_out,
                          color=sector_colors[i], alpha=0.18, zorder=1)
        ax_p.plot(thetas, [r_out]*len(thetas),
                  color=sector_colors[i], linewidth=2, alpha=0.7)
        ax_p.plot(thetas, [r_in]*len(thetas),
                  color=sector_colors[i], linewidth=2, alpha=0.7)
        for th in [thetas[0], thetas[-1]]:
            ax_p.plot([th, th], [r_in, r_out],
                      color=sector_colors[i], linewidth=2, alpha=0.7)
        mid = np.deg2rad(r['hue_center'])
        ax_p.text(mid, r_out + 0.08, r['name'],
                  ha='center', va='center', fontsize=12,
                  fontweight='bold', color=sector_colors[i])

    ax_p.set_ylim(0, 1)
    ax_p.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_p.set_yticklabels([])
    munsell = ['5R', '5YR', '5Y', '5GY', '5G', '5BG', '5B', '5PB', '5P', '5RP']
    angles = np.linspace(0, 2*np.pi, 11)[:-1]
    ax_p.set_xticks(angles)
    ax_p.set_xticklabels([])
    for ang, lab in zip(angles, munsell):
        ax_p.text(ang, 1.13, lab, ha='center', va='center',
                  fontsize=10, fontweight='bold', color='#333')
    ax_p.grid(True, alpha=0.25, linewidth=0.4, color='#999')
    ax_p.spines['polar'].set_color('#bbb')
    ax_p.set_title(f'MCC Range Spec — {result["compliance_pct"]:.1f}% of pixels in range',
                   fontsize=12, pad=20, fontweight='bold')

    # Hue histogram
    ax_h = fig.add_subplot(gs[0, 1])
    ax_h.set_facecolor('#f0f0f0')
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [hsv_to_rgb([c/360, 0.7, 0.85]) for c in centers]
    ax_h.bar(centers, hist, width=HUE_BIN_DEG, color=bar_colors,
             edgecolor='none', align='center')

    for i, r in enumerate(regions):
        hs, he = r['hue_range']
        if hs <= he:
            ax_h.axvspan(hs, he, color=sector_colors[i], alpha=0.18, zorder=0)
        else:
            ax_h.axvspan(hs, 360, color=sector_colors[i], alpha=0.18, zorder=0)
            ax_h.axvspan(0, he, color=sector_colors[i], alpha=0.18, zorder=0)
        ax_h.axvline(r['hue_center'], color=sector_colors[i],
                     linewidth=2, linestyle='--', alpha=0.8)
        ax_h.text(r['hue_center'], hist.max()*0.95, r['name'],
                  ha='center', fontsize=10, fontweight='bold',
                  color=sector_colors[i])
    ax_h.set_xlim(0, 360)
    ax_h.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax_h.set_xlabel('Hue (°)', fontsize=9)
    ax_h.set_yticks([])
    ax_h.set_title('Hue Distribution & Peaks', fontsize=11, pad=8, fontweight='bold')
    for spine in ['top', 'right', 'left']:
        ax_h.spines[spine].set_visible(False)
    ax_h.spines['bottom'].set_color('#888')

    # Spec table
    ax_t = fig.add_subplot(gs[1, 1])
    ax_t.axis('off')
    headers = ['', 'Hue range', 'S range', 'V range', 'Share']
    rows = []
    for i, r in enumerate(regions):
        hs, he = r['hue_range']
        h_str = f'{hs:.0f}°–{he:.0f}°' if hs <= he else f'{hs:.0f}°↻{he:.0f}°'
        rows.append([
            r['name'], h_str,
            f"{r['s_range'][0]:.2f}–{r['s_range'][1]:.2f}",
            f"{r['v_range'][0]:.2f}–{r['v_range'][1]:.2f}",
            f"{result['region_share'][r['name']]:.1f}%",
        ])
    table = ax_t.table(cellText=rows, colLabels=headers,
                       cellLoc='center', loc='center',
                       bbox=[0.0, 0.1, 1.0, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor('#444')
        cell.set_text_props(color='white', weight='bold')
    for i in range(len(rows)):
        table[(i+1, 0)].set_facecolor(sector_colors[i])
        table[(i+1, 0)].set_text_props(weight='bold', color='white')
    ax_t.set_title('MCC Spec Summary', fontsize=11, pad=8, fontweight='bold')

    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='#f0f0f0')
    plt.close(fig)

# =====================================================================
# 4. 스크립트로 실행할 때 — 명동 strip에서 MCC 도출하고 시각화
# =====================================================================
if __name__ == "__main__":
    spec = compute_mcc_spec(IMG_PATH)
    regions = spec['regions']

    # 콘솔 출력
    print("=== MCC Spec ===")
    for r in regions:
        print(f"\n{r['name']}")
        print(f"  Hue center : {r['hue_center']:.1f}° ({to_munsell(r['hue_center'])})")
        print(f"  Hue range  : [{r['hue_range'][0]:.1f}°, {r['hue_range'][1]:.1f}°]")
        print(f"  S range    : [{r['s_range'][0]:.3f}, {r['s_range'][1]:.3f}]")
        print(f"  V range    : [{r['v_range'][0]:.3f}, {r['v_range'][1]:.3f}]")

    # 원본 이미지에 대한 MCC 적합도
    result = check_mcc_image(IMG_PATH, regions)
    print(f"\n=== Compliance on source image ===")
    print(f"  Total chromatic pixels: {result['chromatic_pixels']:,}")
    print(f"  In MCC range: {result['in_mcc_pixels']:,}")
    print(f"  Compliance: {result['compliance_pct']:.1f}%  "
          f"({'COMPLIANT' if result['compliant'] else 'NON-COMPLIANT'})")
    for name, share in result['region_share'].items():
        print(f"    {name}: {share:.1f}%")

    # 시각화
    visualize(spec, result, OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")

    # Demo — 임의 색 한 개 검증
    print("\n=== Demo: single color check ===")
    test_colors = [
        ('Red signage',       (0.02, 0.85, 0.7)),
        ('Blue signage',      (0.6,  0.7,  0.6)),
        ('Bright green sign', (0.33, 0.9,  0.8)),
        ('Pastel pink',       (0.95, 0.3,  0.85)),
    ]
    for name, (h, s, v) in test_colors:
        ok, region = check_mcc(h, s, v, regions)
        verdict = f"OK ({region})" if ok else "REJECTED"
        print(f"  {name:20s} H={h*360:5.1f}° S={s:.2f} V={v:.2f}  → {verdict}")
