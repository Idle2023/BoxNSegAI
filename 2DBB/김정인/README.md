# SSD (Single-shot multibox detection)

### Network
#### 세 가지 네트워크로 구성될 수 있는 CNN (Convolutional Neural Network)
1. Base convolutions
  - ImageNet 에 사전 학습된 VGG16을 사용
  - low level feature map 을 추출
  - Input image size: (300,300)
  - VGG16 의 fc layer (fc6, fc7) 를 conv layer (conv6, conv7) 로 대체함
2. Auxiliary convolutions
  - Base network에 추가적인 conv layer를 더 쌓아서 higher level의 feature map을 추출
  - 중간 중간의 conv layer들에서 feature map을 추출하여 서로 다른 크기와 종횡비의 default boxes를 출력
      - 총 6개의 layer [conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2]에서 [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)] 크기의 feature map 추출
      - feature map에서 추출된 prior 개수: 8732
3. Prediction convolutions
  - localization prediction: prior 에서 예측된 바운딩 박스의 위치를 기반으로 offset(g_c_x, g_c_y, g_w, g_h)를 예측
  - class prediction: 각 location 에서 클래스의 존재 확률을 예측
  - Prediction stage 에서 모델의 최종 출력: offset(shape: 8732x4), class_score(shape: 8732Xclass_num)

### Implementation
#### 실행환경
1. Google Colab
```
📦 2dbb_ji
├─ 2DBBs
│  ├─ training.txt
│  └─ training
│     ├─ images
│     └─ labels
├─ json_files
│  ├─ label_map.json
│  ├─ training_images.json
│  └─ training_objects.json
├─ datasets.py
├─ model.py
├─ utils.py
└─ train.ipynb
```


2. Local
```
python3 train.py
```

