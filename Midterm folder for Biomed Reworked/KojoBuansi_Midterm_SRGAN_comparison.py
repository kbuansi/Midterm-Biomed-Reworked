#!/usr/bin/env python
# coding: utf-8

# In[211]:


import tensorflow as tf

import tensorflow_hub as hub

import numpy as np

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

import os

import cv2

import glob

from torchvision.transforms import Resize

from torch.utils.data import DataLoader

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.models import Model

import PIL

from PIL import Image

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError

from torch.utils.data import Dataset

from torchvision import transforms

from torchvision.transforms import Compose

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import Add

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import PReLU

import matplotlib.pyplot as plt


# In[212]:


from tensorflow.keras.layers import UpSampling2D


# In[213]:


# Directories

path = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1'

train_dir = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\train'

test_dir = r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\test'


# In[214]:


# Set data_dir to path

data_dir = path


# In[215]:


# Defining data transformation 

train_transforms = transforms.Compose([
    
    transforms.Resize((32, 32)),
    
    transforms.RandomHorizontalFlip(),
    
    transforms.RandomVerticalFlip(),
    
    transforms.ColorJitter(brightness=0.9, contrast=0.6, saturation=0.1, hue=0.5),
     
    transforms.ToTensor()
])


# In[216]:


# Defining data transformation for high resolution

high_res_transforms = transforms.Compose([
    
    transforms.Resize((128, 128)),
    
    transforms.ToTensor()
])


# In[217]:


# Defining data transformation for test

test_transforms = transforms.Compose([
    
    transforms.Resize((32, 32)),
    
    transforms.ToTensor(),
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    
                         std=[0.229, 0.224, 0.225])
])


# In[218]:


#Creating Dataset

class MyDataset(Dataset):
    
    def __init__(self, data_dir, transforms=None, high_res_transform=None):
        
        self.data_dir = data_dir
        
        self.transforms = transforms
        
        self.high_res_transform = high_res_transform
        
        self.img_paths = os.listdir(data_dir)

    def __getitem__(self, item):
        
        path = os.path.join(self.data_dir, self.img_paths[item])
        
        img = PIL.Image.open(path).convert('RGB')

        
        img_low = img
        
        img_high = img
        
        if self.transforms is not None:
            
            img_low = self.transforms(img)
        
        if self.high_res_transform is not None:
            
            img_high = self.high_res_transform(img)

        
        # Transpose Image Tensor
        
        img_low = np.transpose(img_low, (1, 2, 0))
        
        img_high = np.transpose(img_high, (1, 2, 0))

        return (img_low, img_high)
 
    def __len__(self):
        
        return len(self.img_paths)


# In[219]:


#Createing Two PyTorch Data Loaders

batch_size = 32

train_dataset = MyDataset(train_dir, transforms=train_transforms, high_res_transform=high_res_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

test_dataset = MyDataset(test_dir, transforms=test_transforms)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[220]:


#Generator Model 

def build_generator(input_shape=(32, 32, 3)):
    
    inputs = Input(input_shape)
    
    
    x = Conv2D(64, kernel_size=9, strides=1, padding='same')(inputs)
    
    x = PReLU()(x)
    
    x1 = x
    
    
    for _ in range(5):
        
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        
        x = BatchNormalization()(x)
        
        x = PReLU()(x)
        
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        
        x = BatchNormalization()(x)
        
        x = Add()([x1, x])
        
        x1 = x

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    
    x = BatchNormalization()(x)
    
    x = Add()([x1, x])

    # Upsampling
    
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    
    x = PReLU()(x)
    
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    
    x = PReLU()(x)

    x = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=x, name='generator')
    
    return model
     


# In[221]:


#Discirminator creation

def build_discriminator(input_shape=(128, 128, 3)):
    
    inputs = Input(input_shape)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    
    x = LeakyReLU(alpha=0.2)(x)

    
    layer_filters = [64, 128, 128, 256, 256, 512, 512]
    
    strides = [2, 1, 2, 1, 2, 1, 2]

    for filters, stride in zip(layer_filters, strides):
        
        x = Conv2D(filters, kernel_size=3, strides=stride, padding='same')(x)
        
        x = BatchNormalization()(x)
        
        x = LeakyReLU(alpha=0.2)(x)

    
    x = Flatten()(x)
    
    x = Dense(1024)(x)
    
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(1, activation='sigmoid')(x)

    
    
    model = Model(inputs=inputs, outputs=x, name='discriminator')
    
    return model


# In[222]:


#Generator and Discriminator model generation

generator = build_generator()

discriminator = build_discriminator()


generator.summary()

discriminator.summary()


# In[223]:


# Build generator and discriminator

generator = build_generator()


# In[224]:


# Build discriminator

discriminator = build_discriminator()


# In[225]:


# Compile discriminator

discriminator.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])


# In[226]:


# Random seed

np.random.seed(0)

tf.random.set_seed(0)


# In[227]:


# Define training loop

epochs = 50

for epoch in range(epochs):
    
    num_batches = len(train_dataset) // batch_size
    
    for i in range(num_batches):
        
        batch_low_res = []
        
        batch_high_res = []
        
        for _ in range(batch_size):
            
            idx = np.random.randint(len(train_dataset))
            
            low_res_img, high_res_img = train_dataset[idx]
            
            
            batch_low_res.append(low_res_img.numpy())
            
            batch_high_res.append(high_res_img.numpy())

        batch_low_res = np.stack(batch_low_res)
        
        batch_high_res = np.stack(batch_high_res)
        
        # Train discriminator
        
        high_res_fake = generator(batch_low_res)
        
        real_labels = np.ones((batch_size, 1))
        
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(batch_high_res, real_labels)
        
        d_loss_fake = discriminator.train_on_batch(high_res_fake, fake_labels)
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        
        real_labels = np.ones((batch_size, 1))
        
        g_loss = srgan.train_on_batch(batch_low_res, real_labels)


# In[228]:


# Load low-resolution image

low_res_transforms = transforms.Compose([
    
    transforms.Resize((32, 32)),
    
    transforms.ToTensor()
])


# In[229]:


# Access dataset

dataset = MyDataset(r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\train\DRUSEN', transforms=low_res_transforms)

low_res_img, _ = dataset[0]


# In[230]:


# Load high-resolution image

high_res_transforms = transforms.Compose([
    
    transforms.Resize((128, 128)),
    
    transforms.ToTensor()
])


# In[231]:


high_res_dataset = MyDataset(r'C:\Users\Kojo PC\Dropbox\PC (3)\Downloads\Data1\train\DRUSEN', transforms=high_res_transforms)

_, high_res_img = high_res_dataset[0]


# In[232]:


# Prepare the low-resolution image for inference

low_res_img = low_res_img.unsqueeze(0) 

low_res_img = low_res_img.numpy()  


# In[233]:


# Upscale images from 32x32 to 128x128

high_res_fake = generator.predict(low_res_img)


# In[234]:


# Image display

low_res_img_display = low_res_img[0].transpose(0, 1, 2)

high_res_fake_display = high_res_fake[0].transpose(0, 1, 2)

high_res_img_display = high_res_img.transpose(2, 0, 1)


# In[235]:


# Graph original image

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.imshow(high_res_img_display)

plt.title('Original OUTPUT (128x128)')

plt.axis('off')


# In[236]:


# Graph 32x32 imagelow resolution

plt.subplot(1, 3, 2)

plt.imshow(low_res_img_display)

plt.title('Low-resolution (32x32)')

plt.axis('off')


# In[237]:


# Graph Upscaled 128x128 high resolution

plt.subplot(1, 3, 3)

plt.imshow(high_res_fake_display)

plt.title('SRGAN High-resolution (128x128)')

plt.axis('off')

plt.show()


# In[ ]:




