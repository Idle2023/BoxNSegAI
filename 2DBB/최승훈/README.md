## Inroduction
자율주행의 핵심은 차량이 주변 환경을 정확하게 인식하고, 그 정보를 바탕으로 안전하게 주행하는 것입니다. 이번 프로젝트에서는 자율주행 차량의 환경 인식 기능을 위한 중요한 데이터셋, 2DBB (2D Bounding Box) 데이터셋을 처리하는 방법을 소개하겠습니다.

2DBB 데이터셋에는 자동차, 트럭, 버스, 특수 차량, 오토바이, 자전거, 보행자, 교통 표지판, 교통 신호등 등 다양한 객체들의 위치와 종류 정보가 포함되어 있습니다. 이러한 정보는 이미지 내에서 객체의 위치를 나타내는 2D 바운딩 박스 (bounding box) 형태로 제공됩니다.

본 문서에서는 CustomDataset을 구현하여 이러한 데이터셋을 효과적으로 처리하고, 데이터 로딩을 위한 DataLoader, 그리고 YOLOv5를 기반으로 한 객체 탐지 모델을 학습하는 방법을 다루겠습니다. 이를 통해, 자율주행 차량의 환경 인식 능력을 향상시킬 수 있는 딥러닝 모델을 구축하고 평가하는 과정을 이해하실 수 있을 것입니다.

## CustomDataset & DataLoader
파일 구조는 다음과 같습니다. 공모전에서 제공한 100000장의 사진과 레이블 중 10000장을 사용했습니다.
cf) training:vaildation:test = 8:1:1
```python
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
### CustomDataset 
**\_\_getitem\_\_ function** 

파이썬의 매직 메서드 중 하나로, 인덱스를 사용하여 데이터셋의 특정 항목을 가져올 때 호출됩니다. 
이 함수는 주어진 인덱스에 대한 이미지 및 해당 레이블 정보를 처리하고 반환합니다.
- 이미지 처리: 해당 인덱스의 이미지 경로를 찾아 이미지를 로드하고, 전처리(transform)를 적용합니다.
- 레이블 정보 로드: 해당 이미지와 연관된 JSON 레이블 파일을 로드합니다. 이 파일에서는 바운딩 박스의 좌표와 각 객체의 레이블 정보를 가져옵니다.
- 바운딩 박스 좌표 조정: 이미지 크기가 바뀌었기 때문에, adjust_box_coordinates 함수를 사용하여 바운딩 박스의 좌표를 적절히 조정합니다.
- 타겟 생성: 각 객체의 레이블과 바운딩 박스 정보를 결합하여 타겟을 생성합니다. (x_min, y_min, w, h) -> (x_cen, y_cen, w, h)
- 리턴: 처리된 이미지, 타겟 정보, 이미지 경로 및 원본 이미지의 크기와 스케일링 정보를 함께 반환합니다.
### DataLoader
**데이터 로딩**

DataLoader는 데이터셋의 항목들을 배치로 자동으로 가져오는 유틸리티입니다. 여기서는 CustomDataset을 사용하여 학습, 검증, 테스트용 데이터 로더를 각각 생성합니다.
* shuffle: 학습 데이터셋에서는 shuffle=True로 설정하여 각 에폭마다 데이터가 무작위 순서로 제공되도록 합니다. 이는 모델이 학습 도중 데이터의 순서에 익숙해지는 것을 방지하고, 일반화 성능을 향상시키는 데 도움이 됩니다. 검증 및 테스트 데이터셋에서는 shuffle=False로 설정하여 데이터 순서가 동일하게 유지되도록 합니다.
* num_workers: 멀티프로세싱을 활용하여 데이터 로딩 속도를 향상시킵니다. 여기서는 8개의 워커 프로세스를 사용하여 데이터를 병렬로 로드합니다.
* collate_fn: 사용자 지정 배치 처리 함수를 DataLoader에 제공합니다.
  
**collate_fn function**
  
- 배치의 각 항목에서 이미지, 레이블, 경로 및 모양 정보를 분리합니다.
- 각 이미지에 대한 레이블 정보에 이미지 인덱스를 추가합니다. 이는 후속 처리에서 해당 레이블이 어떤 이미지에 속하는지 판별하기 위함입니다.
- 모든 이미지를 스택하여 하나의 배치 텐서로 만들고, 모든 레이블도 연결(concatenate)합니다.

## model
mode은 자율주행이기 때문에 비교적 빠른 **YOLOv5**을 사용했습니다.
pytroch로 된 YOLOv5 공식 repository를 clone해서 사용했습니다. 
number of classes을 데이터셋에 맞게 10으로 바꾸어 주었습니다.
크게 5가지 모듈을 import 해서 사용합니다.
* from models.yolo import **DetectionModel**
    * YOLOv5 모델입니다.
    * 이미지를 분석하고 바운딩 박스, 클래스 라벨, 신뢰도 점수와 같은 객체 검출 결과를 생성하도록 설계되었습니다.
* from models.yolo import **Detect**
    * 실제 검출 단계를 담당할 모듈로 보입니다. 여기서 모델의 원시 출력이 처리되어 일관된 바운딩 박스 좌표, 관련 클래스 라벨 및 신뢰도 점수를 생성합니다.
    * 객체 점수 임계값 설정 또는 앵커 박스 조정과 같은 다양한 후처리 단계가 포함될 수 있습니다.
* from utils.loss import **ComputeLoss**
    * 이름에서 알 수 있듯이, 이 유틸리티는 훈련 단계 중 YOLOv5 모델의 손실을 계산합니다.
    * 바운딩 박스 좌표를 위한 지역화 손실, 객체 카테고리를 위한 분류 손실 등 여러 구성 요소를 고려할 수 있습니다.
* from models.common import **Detections**
    * 이 모듈은 시각화를 위해 특별히 설계되었습니다.
    * 모델의 검출 결과가 주어지면, Detections는 이미지 위에 검출된 바운딩 박스를 그리는데 사용될 수 있으며, 클래스 라벨을 오버레이하고 신뢰도 점수를 표시할 수도 있습니다.
    * 또한 다른 클래스에 대한 색상 코딩 또는 박스 두께 및 기타 시각적 파라미터 조정과 같은 작업을 처리할 수도 있습니다.
* from utils.general import **non_max_suppression**
    * 객체 검출에서 필수적인 후처리 단계입니다.
    * 모델이 하나의 객체에 대해 여러 바운딩 박스를 검출할 때, 이 함수는 덜 확실한 박스를 억제(또는 제거)하여 각 객체가 가장 확실한 바운딩 박스만으로 표현되도록 합니다.

* run 함수는 validation 결과를 평가해주는 함수입니다. 공식 홈페이지에서는 inference mode로 바뀌어 버려서 나중에 gradients
를 업데이트 할 수 없게 되어 공식 홈페이지 코드를 정리해 사용했습니다. 데코레이터를 제거해 주었습니다.
```python
@smart_inference_mode()
def run()
```

## train
* 		Fine-tuning 준비
    * 모델의 모든 파라미터들을 훈련 가능하게 설정합니다. 
    * 모델은 GPU 또는 CPU로 이동되며, 이에 따라 연산이 수행됩니다.
* 		손실 계산
    * ComputeLoss를 사용하여 YOLOv5 모델의 손실을 계산합니다.
    * 손실은 모델의 출력과 실제 라벨 간의 차이를 나타내며, 최적화 도구를 사용하여 모델의 파라미터를 업데이트하는 데 사용됩니다.
* 		한 에포크 동안의 훈련
    * 함수 train_one_epoch는 모델을 훈련 모드로 설정한 후, 훈련 데이터셋의 각 배치에 대해 반복합니다.
    * 각 반복에서는 입력 데이터를 모델에 전달하고 출력을 얻습니다.
    * 계산된 손실을 사용하여 그래디언트를 역전파(Backpropagation)하고, 파라미터를 업데이트합니다.
    * 일정한 간격으로 훈련 중의 손실을 출력합니다.
    * 총 **35 epochs** 정도 진행했습니다. 중간에 network error가 나서 다시 진행했습니다.
* 		학습률 스케줄링
    *  학습이 진행됨에 따라 학습률을 감소시켜, 훈련이 더 안정적으로 수행되도록 합니다.
**초기 transfer learning loss보다 초기 fine tuning 방식의 loss가 더 낮아(5.1 > 2.1) fine tuning을 사용했습니다.**

## Validation & Test
### Validation
- Precision: 예측된 Positive 결과 중 실제로 Positive인 비율입니다. 즉, 얼마나 많은 예측이 올바르게 되었는지 나타냅니다. <br>
- Recall: 실제 Positive 중 예측된 Positive의 비율입니다. 즉, 모든 Positive 대상 중 몇 개를 성공적으로 예측했는지 나타냅니다.<br> 
- mAP@0.5: 평균 정밀도를 기반으로 한 평균 정밀도 값으로, IoU(Intersection over Union) 임계값이 0.5일 때 계산됩니다.<br>
- mAP@0.5:0.95: mAP의 평균 값으로 IoU가 0.5에서 0.95까지 0.05씩 증가할 때마다 계산됩니다. 이것은 모델의 성능을 다양한 IoU 임계값에 대해 평가합니다. <br>
- Box Loss: Bounding box의 예측 오차를 나타냅니다. <br>
- Obj Loss: 객체 예측의 오차를 나타냅니다. <br>
- Class Loss: 클래스 예측의 오차를 나타냅니다. <br>
앞의 epochs 30 정도의 데이터가 날라갔지만 😿 전반적으로 성능이 향상되고 있음을 볼 수 있습니다.

## Test
Precision (정밀도): 예측된 Positive 결과 중 실제로 Positive인 비율입니다. 
이 값이 0.6779로 나온 것은, 모델이 Positive로 예측한 결과 중 약 67.79%가 실제로 Positive임을 나타냅니다. 

Recall (재현율): 실제 Positive 중 예측된 Positive의 비율입니다. 이 값이 0.2975인 것은 전체 Positive 샘플 중 약 29.75%만을 올바르게 예측하였음을 나타냅니다. 
아직 학습이 덜 되었음을 알 수 있습니다.

mAP@0.5: 평균 정밀도를 기반으로 한 평균 정밀도 값으로, 
IoU(Intersection over Union) 임계값이 0.5일 때 계산됩니다. 이 값이 0.4852로 나타나는 것은, 
IoU 임계값이 0.5일 때의 평균 정밀도가 약 48.52%임을 나타냅니다.

mAP@0.5:0.95: mAP의 평균 값으로 IoU가 0.5에서 0.95까지 0.05씩 증가할 때마다 계산됩니다. 
이 값이 0.2940로 나타나는 것은, 다양한 IoU 임계값에 대한 평균 정밀도가 약 29.40%임을 나타냅니다.

## Visualization
앞서 언급한 yolov5 공식 repository의 Detections class를 사용했습니다. test dataset image 중 일부입니다.
<img width="402" alt="Screenshot 2023-08-29 at 9 39 41 PM" src="https://github.com/Idle2023/BoxNSegAI/assets/113033780/372eb4e7-2f32-4133-a78f-7e94edca2220">
<img width="402" alt="Screenshot 2023-08-29 at 9 38 45 PM" src="https://github.com/Idle2023/BoxNSegAI/assets/113033780/c15cfa89-2300-4a94-97ea-f3d5325db640">

 


