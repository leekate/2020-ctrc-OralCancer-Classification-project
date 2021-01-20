# 2020-ctrc-OralCancer-Classification-project
# Main Idea
![image](https://user-images.githubusercontent.com/46522501/102291993-a5385780-3f87-11eb-957b-2086fdd33263.png)
이미지의 밝기를 일정하게 통일하고,
edge를 검출해 병변 부위를 검출해 feature로 사용한다면 성능이 좋아지지 않을까?



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
Cancer: 
Precancer: 
Inflammatory: 
Normal: 
TOTAL: 



# 3. train/validation/test
train: validation: test=약 1: 0.4: 0.4의 비율
cancer: 
precancer: 
Inflammatory: 
normal: 



# 4. Resnet
```
filter_size1 = 7
num_filters1 = 64

filter_size2 = 1
num_filters2 = 64

filter_size3 = 3
num_filters3 = 64

filter_size4 = 1
num_filters4 = 256

filter_size5 = 1
num_filters5 = 128

filter_size6 = 3
num_filters6 = 128

filter_size7 = 1
num_filters7 = 512

filter_size8 = 1
num_filters8 = 256

filter_size9 = 3
num_filters9 = 256

filter_size10 = 1
num_filters10 = 1024

filter_size11 = 1
num_filters11 = 512

filter_size12 = 3
num_filters12 = 512

filter_size13 = 1
num_filters13 = 2048

# Fully-connected layer.
# Number of neurons in fully-connected layer.
# convolution layer 전체에 있는 뉴런 수 = 필터 역할
# fc_size = 256             
fc_size = 1000

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 224

batch_size=32
```


# 5. 2차 결과
Confusion Matrix:
