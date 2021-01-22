# 2020-ctrc-OralCancer-Classification-project

# 0. Start
  암세포 분류에 있어 병리과 의사가 암세포를 구별하는 데에 Deep Learning기술을 통한 암세포 사진 분류를 1차로 진행하고 정확도가 낮은 사진들을 추려 한번 더 확인한다면 시간과 피로를 줄일 수 있다는 연구가 있다.  Deep Learning기술을 이용해 암세포 데이터를 다룰 때 최종적으로 하고자 하는 것은 발생하는 정보의 loss를 최소화하고 전처리 과정을 거쳐 노이즈를 제거해 고해상도 의료 이미지를 훈련 영상으로 이용하여 CNN 모델을 학습시키는 것이다.
  
  기본 CNN, VGG16, ResNet과 같이 다양한 딥러닝 모델들을 통해 구강 내 상태를 분석 및 분류하고, 분류에 최적인 딥러닝 모델을 찾기 위해 각 모델 간의 성능을 비교 및 검증을 시행고자 한다. 





# 1. 데이터 원본
Cancer: 410

Precancer: 150

Inflammatory 265

Normal: 1137

TOTAL: 1962



# 2. 데이터 증량
1) 증량 방법:
증량 후 train: validation: test=약 1: 0.4: 0.4의 비율이 되도록


2) 증량 조건:
rescale=1./255,
horizontal_flip=True,
vertical_flip=True,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
zoom_range=0.2

---->

Cancer: 782

Precancer: 288

Inflammatory: 502

Normal: 2047

TOTAL: 3619



# 3. train/validation/test
train: validation: test=약 1: 0.4: 0.4의 비율

cancer: 496

precancer: 184

Inflammatory: 316

normal: 1365



# 3-2. Histogram Equalization





# Results
사용한 두 모델 모두 50%, 60%의 성능이 나왔다. 이는 상당히 낮은 성능으로 전처리 과정이 추가로 필요해보인다. 관려 논문 읽고 적용해보자.
## 4-1. Resnet
<img width="551" alt="스크린샷 2021-01-22 오후 1 00 02" src="https://user-images.githubusercontent.com/46522501/105444736-c3ddec80-5cb1-11eb-8e67-6769d6e49170.png">

## 4-2. VGG16
![vgg](https://user-images.githubusercontent.com/46522501/105444618-86795f00-5cb1-11eb-9066-9ded9b58c1fc.png)





### 예상
![image](https://user-images.githubusercontent.com/46522501/102291993-a5385780-3f87-11eb-957b-2086fdd33263.png)
이미지의 밝기를 일정하게 통일하고,
edge를 검출해 병변 부위를 검출해 feature로 사용한다면 성능이 좋아지지 않을까?



# 5. Final Result
Confusion Matrix:
