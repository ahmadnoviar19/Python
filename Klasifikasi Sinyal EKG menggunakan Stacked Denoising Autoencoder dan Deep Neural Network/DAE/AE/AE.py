# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 09:42:32 2019

@author: Asus
"""

import numpy as np
from keras.layers import Input, Dense 
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

layin = 252
encod = 126

dam = Adam(lr=0.00005)
x_test = np.load('../data/feature_noisy_decoded.npy')

input_img = Input(shape=(layin,))
encoded = Dense(encod, activation='relu', use_bias=True)(input_img)
decoded = Dense(layin, activation='sigmoid',use_bias=True)(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img,encoded)

autoencoder.compile(optimizer=dam, loss='mean_squared_error')

print(x_test.shape)
autoencoder.fit(x_test, x_test, epochs=200,batch_size=32, shuffle=True,verbose=0)
autoencoder.summary()

avg_loss = np.average(autoencoder.history.history['loss'])
print("rata-rata loss training: ",avg_loss)

plt.figure()
plt.plot(autoencoder.history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.savefig('data/plotlossae50.png')
plt.show()

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

autoencoder.save('data/AE.h5')
encoder.save('data/encod.h5')
np.save('data/feature_encoded.npy', encoded_imgs)
np.save('data/feature_decoded.npy', decoded_imgs)

with open('data/hasilae.txt', 'w') as tes:
    tes.write('rata-rata loss {0} \n'.format(str(avg_loss)))