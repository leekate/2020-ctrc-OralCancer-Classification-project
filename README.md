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



# 4-1. Resnet
```

from tensorflow.keras.applications import ResNet50

conv_base=ResNet50(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))
                
                
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import optimizers, initializers, regularizers, metrics



model=models.Sequential()
model.add(conv_base)



model.add(layers.Dense(2048, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(2048, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add( MaxPooling2D((2,2), padding='same'))




model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))


model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))


model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))

model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))


model.add(Flatten()) 
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(4,activation='relu'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```



# 4-2. VGG16
```
from tensorflow.keras.applications import VGG16

conv_base=VGG16(weights='imagenet',
                include_top=False,
                input_shape=(150,150,3))
                
                
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import optimizers, initializers, regularizers, metrics


model=models.Sequential()
model.add(conv_base)


model.add(layers.Dense(2048, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(2048, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add( MaxPooling2D((2,2), padding='same'))



#Layer weight regularizers    #kernel_regularizer: Regularizer to apply a penalty on the layer's kerne 
#L1: float; L1 regularization factor = 0.001   
#L1 regularization : 가중치의 절댓값에 비례하는 비용이 추가됨(가중치의 L1 norm)
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add( MaxPooling2D((2,2), padding='same'))



  
#L2: float; L2 regularization factor = 0.001  
#L2 regularization(=weight decay) : 가중치의 제곱에 비례하는 비용이 추가됨(가중치의 L2 norm)
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(512, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add( MaxPooling2D((2,2), padding='same'))


model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add( MaxPooling2D((2,2), padding='same'))


model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dense(128, kernel_regularizer = regularizers.l2
                                  (0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add( MaxPooling2D((2,2), padding='same'))



model.add(Flatten()) 
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(4,activation='relu'))



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
```       




# 5. 2차 결과
Confusion Matrix:
