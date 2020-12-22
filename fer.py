import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes=5
img_rows,img_colms=48,48
size=8
train_data=r'E:\project\facial expression detection\face-expression-recognition-dataset\images\train'
valid_data=r'E:\project\facial expression detection\face-expression-recognition-dataset\images\validation'
train_data_gen=ImageDataGenerator(rescale=1./255,rotation_range=30,shear_range=0.3,zoom_range=0.3,width_shift_range=0.4,height_shift_range=0.4,horizontal_flip=True,vertical_flip=True)

valid_data_gen =ImageDataGenerator(rescale=1./255)

train_generator=train_data_gen.flow_from_directory(train_data,color_mode='grayscale',target_size=(img_rows,img_colms),batch_size=size,class_mode='categorical',shuffle=True)

validation_generator=valid_data_gen.flow_from_directory(valid_data,color_mode='grayscale',target_size=(img_rows,img_colms),batch_size=size,class_mode='categorical',shuffle=True)
 
model= Sequential()
#block 1
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2*2))
model.add(Dropout(0.2))

#block2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block3
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#block4
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1))) 
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_colms,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

 #block5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#block7
model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint=ModelCheckpoint(r'E:\project\facial expression detection\Emotional_little_vgg.h5',monitor='val_loss',mod='min',save_best_only=True,verbose=1)

earlystop=EarlyStopping(monitor='val_loss',min_delta=0,patience=3,verbose=1,restore_best_weights=True)

reduce_rate=ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,varbose=0,min_delta=0.0001)

callback=[checkpoint,earlystop,reduce_rate]

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

nb_train_samples=24282
nb_validation_sample=5937
epoch=25

history=model.fit_generator(train_generator,steps_per_epoch=nb_train_samples//size,epochs=epoch,callbacks=callback,validation_data=validation_generator,validation_steps=nb_validation_sample//size )

