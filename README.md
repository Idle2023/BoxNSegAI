# 2DBB Object‐Detection Pipeline (YOLOv5)

> **🏆 2023 인공지능 자율주행 알고리즘 챌린지 – 장려상**


## 소개

본 프로젝트는 도로 위에서 주변 객체를 빠르고 정확하게 인식하는 것이 목표입니다. 2D Bounding Box(2DBB) 데이터셋을 활용해 YOLOv5를 Fine-Tuning 하여 차량, 보행자, 신호등 등 총 9종의 Object를 Detection하고, 시각화하였습니다.


## 데이터셋 요약
| 항목 | 값 |
|------|----|
| 이미지/라벨 수 | 100 000 / 100 000 |
| Split | train : val : test = **8 : 1 : 1** |
| 포맷 | COCO json (`bbox`) |
| 클래스 | car, truck, bus, special_vehicle, motorcycle, bicycle, pedestrian, traffic_sign, traffic_light, none |
| 비고 | `none` 클래스는 평가 제외 |

### 폴더 구조
```
2DBB/
│ # images/ .jpg (1920 x 1080)
| # labels/ .json
├── training/
│   ├── images/                
│   └── labels/                
├── validation/
│   ├── images/                
│   └── labels/                
└── test/
    ├── images/                
    └── labels/
```


## 파이프라인 개요
### CustomDataset & DataLoader
* `__getitem__` 단계
  1. 이미지 로드 후 변환(transform) 적용
  2. 대응 JSON 파싱 → bbox 좌표 보정(`adjust_box_coordinates`)
  3. (x_min, y_min, w, h) → (x_cen, y_cen, w, h) 변환
* `collate_fn`
  * 가변 길이 bbox를 concat, 이미지 스택 → 하나의 Batch Tensor 생성

### YOLOv5 핵심 수정 사항
| 위치 | 변경 내역 |
|------|----------|
| `models/yolov5s.yaml` | `nc: 80 → 10` (클래스 수 교체) |
| `models/yolo.py::Detect.forward` | 앵커 Auto‑shape 사용 해제(고정 앵커) |
| `train.py` | `@smart_inference_mode` 데코레이터 제거 → **`run()`** 함수에서 gradient 흐름 유지 |
| `loss.py` | λ_box, λ_obj, λ_cls 하이퍼파라미터 재조정(1.0, 0.4, 0.4) |
| `utils/plots.py` | 한글 class 이름 표시를 위해 `matplotlib.rcParams["font.family"] = "NanumGothic"` |

> 🔧 **Tip**: 헤더 교체 시 pretrained 백본을 유지하려면 `load_state_dict(strict=False)` 로 가중치를 불러와야 합니다.

### Training vs Evaluation 동작
```text
# model.train() 출력 (3‑scale feature map)
[1, 3, 80, 80, 15]
[1, 3, 40, 40, 15]
[1, 3, 20, 20, 15]
    └── 1 batch × 3 anchors × (H×W) × (4 bbox + 1 obj + 10 cls)

# model.eval() 출력
result[0] : [1, 25 200, 15]   # NMS 전 예측
result[1] : [리스트]           # 학습 모드와 동일한 3‑scale feature map
```

### 학습 루프 하이라이트
1. **Optimizer** : AdamW (lr 1e‑4 → 1e‑6 Cosine)
2. **Epochs** : 35 (중간 네트워크 오류로 재개)
3. **Loss** : `ComputeLoss` – Box·Obj·Cls
4. **Mixed Precision** : `torch.cuda.amp` 적용, VRAM ≈ 6 GB


## 성능
| Metric | Val | Test |
|--------|-----|------|
| Precision | 0.71 | 0.68 |
| Recall    | 0.45 | 0.30 |
| mAP@0.5   | 0.56 | 0.49 |
| mAP@0.5:0.95 | 0.32 | 0.29 |

*본 실험은 100,000장 이미지 중 10,000장으로 진행되었습니다.


## 시각화
<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/848a3b13-9ffe-4475-a500-370432349a68" width="49%">
  <img src="https://github.com/user-attachments/assets/0f645c3d-941f-4ae5-97d6-2fb3b27aa783" width="49%">
</div>


## 수상 내역
| 구분 | 상금       | 팀 수 |
|------|----------|------|
| 최우수상 | 300만원    | 1팀 |
| 우수상   | 200만원   | 1팀 |
| **장려상** | **100만원** | **2팀** |

본 프로젝트에서는 2D Bounding Box의 mAP 성능이 우수하게 나타났습니다.


## 참고
* [YOLOv5 ‑ Ultralytics](https://github.com/ultralytics/yolov5)
* [공모전 데이터 가이드](https://challenge2023.gcontest.co.kr/template/m/frame/downloadlist/12709?q=617)

