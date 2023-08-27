# BoxNSegAI
2023 자율주행 인공지능 알고리즘 개발 챌린지
## 코드 짜는 순서
1. Pre-trained model 고르기 model class number 바꿔주기!!
2. input output shape print하기!!
3. dataset loader 만들기
4. Layer fine tuning 마지막 몇 개의 layer만(loss function, learning rate, optimizer 조정, overfitting 안 되도록 조정)
5. Post processing 필수 
6. 2DBB, 3DBB mAP 높도록, 2DSS mIoU 높도록

**중간 중간 결과 확인할 수 있게 시각화 또는 print 꼭 하기!!**

### yolov5(2DBB)
Training Mode (model.train())
len(result_train): 3
YOLO는 3 개의 다른 크기의 특징 맵을 사용하여 탐지를 수행합니다. 각 특징 맵은 다른 스케일의 객체를 탐지하는 데 사용됩니다.
각 특징 맵의 형태:
torch.Size([1, 3, 80, 80, 15])
torch.Size([1, 3, 40, 40, 15])
torch.Size([1, 3, 20, 20, 15])
여기서:
- 1: 배치 크기
- 3: 각 특징 맵에 대한 기본 박스(anchors)의 수
80x80, 40x40, 20x20: 각각의 특징 맵의 spatial resolution (공간 해상도)
- 15: 각 기본 박스 당 예측된 속성의 수. 일반적으로 (x, y, width, height, objectness) + 각 클래스에 대한 확률. 예를 들어, 10개의 클래스를 가진 경우 15 = 4 + 1 + 10.

Evaluation Mode (model.eval())
len(result_eval): 2
학습 모드와는 달리 평가 모드에서는 모델이 탐지된 바운딩 박스와 특징 맵 예측 모두를 반환합니다.
result_eval[0].shape: torch.Size([1, 25200, 15])
여기서:
- 1: 배치 크기
- 25200: 전체 특징 맵에서 예측된 바운딩 박스의 총 수 (모든 스케일과 기본 박스에 대한 예측을 합친 것)
- 15: 각 바운딩 박스의 속성 수.

result_eval[1]: 여러 특징 맵의 출력을 포함하는 리스트. 학습 모드의 출력과 동일한 형식을 가지고 있습니다.

### DeepLabv3(2DSS)
- 4: 배치 크기를 나타냅니다. 각 배치에는 4개의 이미지가 있습니다.
- 26: 분할 작업의 클래스 수입니다. 여기의 각 채널은 이미지에 대한 클래스의 확률 맵에 해당합니다. 이는 이미지를 26개의 서로 다른 카테고리 또는 클래스로 분류한다는 의미입니다.
- 513 x 513: 입력 이미지 크기와 일치하는 출력 분할 맵의 높이와 너비입니다. 

train과 eval은 shape이 같지만 값이 다릅니다. train에서는 batchNorm과 Dropout을 수행하기 때문입니다.
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
