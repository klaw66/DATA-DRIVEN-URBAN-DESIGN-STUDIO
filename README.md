# Myeongdong Color Code (MCC)

> 명동 간판 색채 정체성을 정량 측정하고, 지구단위계획 규제 framework으로 번역하는 학부 프로젝트.

명동의 고채도·다색 간판이 단순한 "시각공해"가 아닌 **도시 정체성**이라는 가설에서 출발.
스트리트뷰 facade 이미지를 HSV 색공간으로 분석해 명동의 **주색 두 개(MCC-1, MCC-2)** 를 데이터 기반으로 정의하고,
신규 간판이 이 범위 안에 들어오는지 자동 판정한다.

---

## 📦 설치

```bash
pip install -r requirements.txt
```

또는 분석만 필요하면 가벼운 설치:
```bash
pip install numpy matplotlib opencv-python scipy
```

세그멘테이션까지 필요하면 PyTorch 추가:
```bash
pip install torch torchvision segmentation-models-pytorch
```

---

## 🧠 Pre-trained 모델 가중치 (segmentation용)

`infer_to_hsv.py`는 학습된 간판 세그멘테이션 모델 가중치 파일이 필요하다:
```
best_model_deeplabv3plus_resnet50.pth
```

이 파일은 git에 포함되지 않는다 (용량 큼). Kumakoshi et al. (2021) 방법론 기반으로
별도 학습한 가중치를 프로젝트 루트에 두면 된다.

> 가중치 없이도 `hsv_analysis_batch.py`, `mcc_range.py`는 정상 동작한다 —
> 미리 마스킹한 간판 이미지를 input으로 쓰면 됨.

---

## 🛠️ 스크립트 3종

### 1. `hsv_analysis_batch.py` — 폴더 일괄 HSV 분석

`input_images/` 폴더의 모든 이미지를 한꺼번에 분석. Macau paper(Chen et al. 2022) 스타일.

```bash
# 1. input_images/ 폴더에 이미지들 넣기
# 2. 실행
python hsv_analysis_batch.py
```
**출력**:
- `output_results/<이미지명>_hsv.png` — 각 이미지별 polar plot(H) + S/V 히스토그램
- `output_results/summary_stats.csv` — 전체 통계 요약 (Excel로 비교용)

### 2. `infer_to_hsv.py` — 간판 세그멘테이션 + HSV 분석 ⭐

**핵심 파이프라인**: 딥러닝 모델로 간판 영역만 분리한 뒤, 그 영역의 픽셀만 HSV 분석.
가로수·차량·하늘 등 비-간판 픽셀이 분석에 섞이지 않으므로 **명동 간판의 진짜 색채 DNA**가 드러남.

```bash
# 1. best_model_deeplabv3plus_resnet50.pth 를 프로젝트 루트에 두기
# 2. input_images/ 폴더에 facade 이미지 넣기
# 3. 실행
python infer_to_hsv.py
```
**출력**: 각 이미지마다 `_mask.png`, `_overlay.jpg`, `_hsv.png` 자동 생성 + `summary_stats.csv`
(`summary_stats.csv`에는 간판 면적비 `sign_area_pct`도 포함됨)

### 3. `mcc_range.py` — MCC 주색 범위 도출 + 적합성 검증

Hue histogram에서 자동으로 두 peak를 찾아 주색 범위로 정의하고, 신규 간판의 MCC 적합성 판정.

```bash
python mcc_range.py
```

다른 코드에서 함수만 import해서 쓰기:
```python
from mcc_range import compute_mcc_spec, check_mcc, check_mcc_image

# 1. 명동 데이터로 MCC 범위 한 번만 도출
spec = compute_mcc_spec("myeongdong_strip.png")
regions = spec['regions']

# 2-A. 새 간판 이미지 적합 여부 판정
result = check_mcc_image("new_signage.jpg", regions)
print(result['compliant'], result['compliance_pct'])

# 2-B. 단일 색(H,S,V) 한 점 판정
ok, region = check_mcc(h_norm=0.05, s=0.8, v=0.7, regions=regions)
```

---

## 🔄 권장 워크플로우

```
1. 명동 스트리트뷰 facade 이미지 수집 → input_images/
       ↓
2. infer_to_hsv.py
   → 간판 mask + 간판 영역만의 HSV 분석
       ↓
3. mcc_range.py
   → 명동 주색 범위 (MCC-1, MCC-2) 자동 도출
       ↓
4. check_mcc_image() 함수
   → 신규 간판 디자인 적합성 평가
```

`hsv_analysis_batch.py`는 segmentation 없이 전체 이미지를 빠르게 훑어볼 때 쓴다.

---

## 📁 폴더 구조

```
myeongdong-color-code/
├── README.md
├── requirements.txt
├── .gitignore
├── hsv_analysis_batch.py      # 폴더 일괄 HSV
├── infer_to_hsv.py            # 간판 segmentation + HSV
├── mcc_range.py               # MCC framework + 검증 함수
├── best_model_deeplabv3plus_resnet50.pth  # (gitignored, 별도 준비)
├── input_images/              # (gitignored) 분석 대상 이미지
└── output_results/            # (gitignored) 결과 출력
```

---

## 🎯 MCC framework 핵심 개념

| 항목 | 정의 |
|---|---|
| **주색 도출** | Hue 히스토그램에서 가장 두드러진 두 peak 자동 식별 |
| **MCC-1, MCC-2** | 각 peak 중심 ± 50° 폭의 Hue 범위 |
| **S/V 하한** | 명동 데이터의 percentile 기반 (회색·그림자 제외) |
| **S/V 상한** | **없음** — 명동 정체성은 고채도·고명도 허용 |
| **적합 판정** | 간판 chromatic 픽셀의 70% 이상이 MCC 범위 안 → 적합 |

> 채도·명도에 **상한을 두지 않는 게 핵심**.
> 이 설계는 "채도를 줄여라"는 기존 시각공해 규제 패러다임(Portella 2014)과 반대 방향으로,
> 명동의 고채도·고명도 풍경을 **보존해야 할 정체성**으로 본다.

---

## 📚 주요 참고문헌

- **Chen et al. (2022).** Computer vision quantization research on the architectural color of Avenida de Almeida Ribeiro in Macau. *PLoS ONE*. — HSV+K-means 방법론
- **Kumakoshi et al. (2021).** Quantifying urban streetscapes with deep learning. *arXiv:2106.15361*. — 간판-입면 segmentation 모델
- **박한나 외 (2021).** 간판의 색채조화와 가독성에 대한 간판색 수의 영향. *디자인학연구*. — 색 수 임계 (2색 = 최적)
- **안새얀 외 (2024).** 입면 차폐도 분석 자동화. *도시설계*, 25(5). — 한국 도시설계 자동화 lineage
- **Visual complexity and preference (2025).** 경리단길 보행자 선호도 연구. *International Journal of Urban Sciences*. — 25–35% 면적비 임계
- **Valdez & Mehrabian (1994).** Effects of color on emotions. *JEP:General*. — 채도-감정 회귀식

---

## 📝 License

학부 프로젝트 · 자유 사용 가능
