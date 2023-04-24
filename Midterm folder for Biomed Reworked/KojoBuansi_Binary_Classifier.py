#!/usr/bin/env python
# coding: utf-8

# In[43]:


import tensorflow as tf


# In[44]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[45]:


from tensorflow.keras.applications import VGG16


# In[46]:


from tensorflow.keras.models import Sequential


# In[47]:


from tensorflow.keras.layers import Dense, Flatten, Dropout


# In[48]:


from tensorflow.keras.optimizers import Adam


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


from sklearn.metrics import confusion_matrix


# In[51]:


import numpy as np


# In[52]:


train_dir = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\train'
val_dir = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\test'


# In[53]:


batch_size = 16
num_epochs = 10
learning_rate = 0.0001


# In[54]:


input_shape = (128, 128, 3)


# In[55]:


vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


# In[56]:


vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


# In[57]:


model = Sequential()
model.add(vgg)


# In[58]:


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[59]:


opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[60]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)


# In[61]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen


# In[62]:


val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)


# In[63]:


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size
)

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[64]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[65]:


val_data = val_datagen.flow_from_directory(
    
    val_dir,
    
    target_size=(128, 128),
    
    batch_size=1,
    
    class_mode='binary',
    
    shuffle=False
)

y_true = val_data.labels
y_pred = model.predict(val_data)
y_pred = np.round(y_pred)

cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xticks([0,1])
plt.yticks([0,1])
plt.xlabel('Predicted label')
plt.ylabel('True label')

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.show()

