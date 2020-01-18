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

x_test = np.load('data/feature.npy')
dam = Adam(lr=0.0001)

input_img = Input(shape=(layin,))
encoded = Dense(encod, activation='relu', use_bias=False)(input_img)
decoded = Dense(layin, activation='sigmoid',use_bias=False)(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img,encoded)

encoded_input = Input(shape=(encod,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input,decoder_layer(encoded_input))

autoencoder.compile(optimizer=dam, loss='mean_squared_error')

print(x_test.shape)
autoencoder.fit(x_test,x_test, epochs=200,batch_size=64, shuffle=True,verbose=1)

autoencoder.summary()

avg_loss = np.average(autoencoder.history.history['loss'])
print("rata-rata loss training: ",avg_loss)

plt.figure()
plt.plot(autoencoder.history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.savefig('data/plotlossae.png')
plt.show()

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

autoencoder.save('AE.h5')
encoder.save('encod.h5')
encoder.save('decod.h5')
np.save('data/feature_encoded.npy', encoded_imgs)
np.save('data/feature_decoded.npy', decoded_imgs)