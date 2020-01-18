# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 09:42:32 2019

@author: Asus
"""

import numpy as np
import math as mat
import matplotlib.pyplot as plt
from keras.layers import Input, Dense 
from keras.models import Model
from keras.optimizers import Adam

layin = 252
encod = 252

dam = Adam(lr=0.00095)
x_train = np.load('../data/feature_noisy.npy')
x_test = np.load('../data/feature.npy')
datnoi = np.ravel(x_train)
datrea = np.ravel(x_test)
snr1 = 'SNR sinyal target untuk Denoising Autoencoder' 
snr2 = 'SNR sinyal rekonstruksi Denoising Autoencoder'

a = np.sum(datrea) ** 2
b = (np.sum(datnoi)-np.sum(datrea)) ** 2
c = 10*mat.log10(a/b)
print(snr1,c)

input_img = Input(shape=(layin,))
encoded = Dense(encod, activation='relu', use_bias=True)(input_img)
decoded = Dense(layin, activation='sigmoid',use_bias=True)(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img,encoded)

encoded_input = Input(shape=(encod,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer=dam, loss='mean_squared_error')

print(x_train.shape)
print(x_test.shape)
autoencoder.fit(x_train, x_test, epochs=400,batch_size=64, shuffle=True,verbose=0)

autoencoder.summary()

avg_loss = np.average(autoencoder.history.history['loss'])
print("rata-rata loss training: ",avg_loss)

plt.figure()
plt.plot(autoencoder.history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')    
plt.savefig('data/plotlossdae.png')
plt.show()

encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)

datrec = np.ravel(decoded_imgs)

d = np.sum(datrec) ** 2
e = (np.sum(datnoi)-np.sum(datrec)) ** 2
f = 10*mat.log10(d/e)
print(snr2,f)

autoencoder.save('data/DAE.h5')
encoder.save('data/encod.h5')
encoder.save('data/decod.h5')
np.save('data/feature_noisy_encoded.npy', encoded_imgs)
np.save('data/feature_noisy_decoded.npy', decoded_imgs)

with open('data/snr.txt', 'w') as tes:
    tes.write('{0} {1} \n'.format(snr1,str(c)))
    tes.write('rata-rata loss {0} \n'.format(str(avg_loss)))
    tes.write('{0} {1}'.format(snr2,str(f)))