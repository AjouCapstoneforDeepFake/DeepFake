Deepfake 탐지 기능에 사용된 모델은 pytorch에서 제공하는 사전학습된 EfficientNet-V2 S size 모델입니다.

모델 학습은 Google Colab에서 진행되었습니다. 
알맞은 경로를 설정하여 사용하시기 바랍니다. 

custom dataset에 맞게 하이퍼파라미터 값들을 조정하여 사용하시기 바랍니다. 


Train_dataset
https://drive.google.com/drive/folders/169_FdHIdCYYDcsOhB7Uh_0mSCj6LeNnw?usp=sharing



## 소스코드 실행 방법 ## 
1. Train_dataset folder 안에 real과 fake 폴더로 구분지어진 상태로 Train_dataset을 준비합니다. 
Train_dataset/real 
Train_dataset/fake 

2. Goolge Colab에서 Google Drive를 mount합니다.

3. 그 외에 코드를 작동하는 데에 필요한 라이브러리를 설치하고 import 해줍니다.

4. 본인에게 맞는 경로를 작성하여 코드를 작동시켜야 합니다. 설정해야하는 경로를 다 표시해두었습니다.  

5. 가중치 파일이 업로드 되면 그 파일을 사용하여 deepfake_image_detection.py, deepfake_video_detection.py 파일에서 가중치 파일 경로와 이미지 파일 경로를 설정하여 작동하면 됩니다. 



