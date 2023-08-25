# BoxNSegAI
2023 자율주행 인공지능 알고리즘 개발 챌린지
## 코드 짜는 순서
1. Pre-trained model 고르기
2. 입력과 출력 맞추기(데이터 전처리) ex) 1920 x 1080 -> 244 x 244 -> predicted -> 1920 x 1080
3. Layer fine tuning 마지막 몇 개의 layer만(loss function, learning rate, optimizer 조정, overfitting 안 되도록 조정)
4. Post processing 필수 
5. 2DBB, 3DBB mAP 높도록, 2DSS mIoU 높도록

**초안은 3번, 4번 일반적으로 쓰는 방식 쓰고 precision mAP or mIoU 계산하는 것까지**

**중간 중간 결과 확인할 수 있게 시각화 또는 print 꼭 하기!!**
### 코드 짤 때 꼭 출력해보기!!
1. model num_classes 확인하기!! pretrained 모델 load하는 방식은 모델마다 다릅니다. 하지만 num_classes을 꼭 확인해야 합니다!!!
```python
# YOLOv5 모델에서 Detect 레이어의 출력 채널 수를 확인하여 클래스 수를 확인합니다.
def get_num_classes_from_detect_layer(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            # (num_anchors * (5 + num_classes)) = out_channels
            # YOLO에서는 각 bounding box당 5개의 값 (x, y, w, h, confidence) 및 클래스별 확률을 가집니다.
            out_channels = module.out_channels
            if out_channels % 3 == 0:  # 3개의 앵커를 감안
                return (out_channels // 3) - 5
    return -1

# Create a new YOLO model with 10 classes
model = DetectionModel(cfg='/content/drive/MyDrive/models/setto10.yaml', nc=10)
print('Before update:', get_num_classes_from_detect_layer(model))

# Load pretrained weights
pretrained_weights = torch.load('yolov5l6.pt')
model_dict = model.state_dict()
# Update the model's weights with the pretrained weights, but skip mismatched layers
pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.info()
print('After update:', get_num_classes_from_detect_layer(model))
```

![image](https://github.com/Idle2023/BoxNSegAI/assets/113033780/6b5ee556-00f0-4490-b1f4-64ca37327513)

2. model input, output 확인하기
```python
img, targets, img_path, shapes = train_dataset[3]
model.train()
img = img.unsqueeze(0)
result_train = model(img)
print(len(result_train), result_train[0].shape, result_train[1].shape, result_train[2].shape)
model.eval()
result_eval = model(img)
print(len(result_eval))
print(result_eval[0].shape)
print(len(result_eval[1]))
print(result_eval[1][0].shape, result_eval[1][1].shape, result_eval[1][2].shape)
```
dataset, loader를 먼저 input 형식에 맞추어야 합니다. __get_item__ 함수를 수정해야 합니다!!
그리고 input(img)를 넣고 ouput(train, eval)를 확인해야 합니다!!!
![image](https://github.com/Idle2023/BoxNSegAI/assets/113033780/f3ae3f0b-6fa8-441e-ae6c-c537e5d1adda)
### yolov5 예시

Training Mode (model.train())

len(result_train): 3
YOLO는 3 개의 다른 크기의 특징 맵을 사용하여 탐지를 수행합니다. 각 특징 맵은 다른 스케일의 객체를 탐지하는 데 사용됩니다.
각 특징 맵의 형태:
torch.Size([1, 3, 80, 80, 15])
torch.Size([1, 3, 40, 40, 15])
torch.Size([1, 3, 20, 20, 15])
여기서:
1: 배치 크기
3: 각 특징 맵에 대한 기본 박스(anchors)의 수
80x80, 40x40, 20x20: 각각의 특징 맵의 spatial resolution (공간 해상도)
15: 각 기본 박스 당 예측된 속성의 수. 일반적으로 (x, y, width, height, objectness) + 각 클래스에 대한 확률. 예를 들어, 10개의 클래스를 가진 경우 15 = 4 + 1 + 10.

Evaluation Mode (model.eval())

len(result_eval): 2
학습 모드와는 달리 평가 모드에서는 모델이 탐지된 바운딩 박스와 특징 맵 예측 모두를 반환합니다.
result_eval[0].shape: torch.Size([1, 25200, 15])
여기서:
1: 배치 크기
25200: 전체 특징 맵에서 예측된 바운딩 박스의 총 수 (모든 스케일과 기본 박스에 대한 예측을 합친 것)
15: 각 바운딩 박스의 속성 수.
result_eval[1]: 여러 특징 맵의 출력을 포함하는 리스트. 학습 모드의 출력과 동일한 형식을 가지고 있습니다.
## colab 
개인 레포지토리는 바로 사본이 저장이 되지만 그룹 레포지토리는 안 됩니다. 😢😢
1. repository를 clone합니다.
1. colab -> file -> download -> .ipynb
1. clone한 폴더에 해당 파일을 옮기고 commit, push합니다. clone 안 하고도 commit 가능!!

## 폴더명, 파일명 규칙
각자 2DBB, 2DSS, 3DBB 폴더에 이름으로 폴더를 하나 만듭니다.
add file -> create new file -> BoxNSegAI/2DBB/최승훈/1.txt
폴더가 생성되려면 최소 하나의 파일이 있어야하므로 임시로 파일을 만들어주고 다른 파일을 올린 뒤에 1.txt를 삭제합니다.
clone 안 하고 진행하려면 폴더를 만들고 폴더에 들어가서 upload file을 눌러서 upload하시면 됩니다.
파일명은 2DBB/최승훈/2DBBs_1.ipynb 이런 식으로 자유롭게 작성해주시면 됩니다!!

## commit message 규칙
- feat : 새로운 기능 추가
- fix : 버그 수정
- docs : 문서 관련
- style : 스타일 변경 (포매팅 수정, 들여쓰기 추가, …)
- refactor : 코드 리팩토링
- test : 테스트 관련 코드
- build : 빌드 관련 파일 수정
- ci : CI 설정 파일 수정
- perf : 성능 개선
- chore : 그 외 자잘한 수정
  
[Angular commit convention](https://velog.io/@outstandingboy/Git-%EC%BB%A4%EB%B0%8B-%EB%A9%94%EC%8B%9C%EC%A7%80-%EA%B7%9C%EC%95%BD-%EC%A0%95%EB%A6%AC-the-AngularJS-commit-conventions)
