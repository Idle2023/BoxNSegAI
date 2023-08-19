# BoxNSegAI
2023 자율주행 인공지능 알고리즘 개발 챌린지
## 코드 짜는 순서
1. Pre-trained model 고르기
2. 입력과 출력 맞추기(데이터 전처리) ex) 1920 x 1080 -> 244 x 244 -> predicted -> 1920 x 1080
3. Layer fine tuning 마지막 몇 개의 layer만(loss function, learning rate, optimizer 조정, overfitting 안 되도록 조정)
4. Post processing 필수 
5. 2DBB, 3DBB mAP 높도록, 2DSS IoU 높도록

**초안은 3번, 4번 일반적으로 쓰는 방식 쓰고 precision mAP or IoU 계산하는 코드**

**중간 중간 결과 확인할 수 있게 시각화 또는 print 꼭 하기!!**
### 의견...
2번이 type error, index error(dimension), 등 어려움 tensor type이 낯설어서 더 어렵당…
Pertrained model에 따라 난이도가 달라짐
library에 없는 경우 GitHub clone해서 사용해야 하는데 어려움…
clone해서 사용하면 paper with code에서 논문 베이스인 거 확인 가능하고 최신 논문 사용 가능하긴 함…..

2DBBs가 1시간 정도 걸림 ㅠㅠ 3DBB 전체 학습시키는 건 포기하는 게 맞는 듯
일단 데이터셋 작은 거 다 올렸으니까 코드 완성하고 그 다음에 정확도 높도록 고치는 게 맞는 듯
그리고 시간 남으면 원래 데이터셋에서 돌려보기!!

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
