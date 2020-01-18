# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 09:39:34 2019

@author: Asus
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle

encod = 126
hidden = 100
layout = 1

validasi =[]

dam = Adam(lr=0.00005)
fitur = np.load('data/feature_encoded.npy')
label = np.load('data/label.npy')
 
training_fitur, validasi_fitur, training_label, validasi_label = train_test_split(fitur, label, test_size = 0.1, stratify=(label))
validasi.append(validasi_fitur)
validasi.append(validasi_label)

training_feature, testing_feature, training_labels, testing_labels = train_test_split(training_fitur, training_label, test_size = 0.2, stratify=(training_label))
validasi.append(training_feature)
validasi.append(testing_feature)
validasi.append(training_labels)
validasi.append(testing_labels)

model = Sequential()
model.add(Dense(units = hidden, input_shape=(encod, ), use_bias=False, activation='relu'))
model.add(Dense(units = hidden, use_bias=False, activation='relu'))
model.add(Dense(units = hidden, use_bias=False, activation='relu'))
model.add(Dense(units = layout, use_bias=False, activation='sigmoid'))
model.compile(optimizer=dam, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(training_feature, training_labels, epochs=150, batch_size=48, shuffle=True, validation_data=(testing_feature, testing_labels))
model.summary()
validasi.append(model)
print("\n")

 
rt_loss_train = np.average(model.history.history['loss'])
validasi.append(rt_loss_train)
rt_loss_test = np.average(model.history.history['val_loss'])
validasi.append(rt_loss_test)
rt_acc_train = np.average(model.history.history['acc'])
validasi.append(rt_acc_train)
rt_acc_test = np.average(model.history.history['val_acc'])
validasi.append(rt_acc_test)
max_acc_train = np.max(model.history.history['acc'])
max_acc_test = np.max(model.history.history['val_acc'])

print("rata-rata loss training: ",  rt_loss_train)
print("rata-rata loss testing: ",  rt_loss_test)
print("rata-rata acc training: %.2f%%" % (rt_acc_train*100))
print("rata-rata acc testing: %.2f%%" % (rt_acc_test*100),"\n")
print("maks acc training: %.2f%%" % (max_acc_train*100))
print("mask acc testing: %.2f%%" % (max_acc_test*100),"\n")

plt.figure()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.savefig('loss.png')
plt.show()

plt.figure()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.savefig('acc.png')
plt.show()

validasi_accuracy = model.evaluate(validasi_fitur, validasi_label)
print("%s: %.2f%%" % (model.metrics_names[1], validasi_accuracy[1]*100),"\n")
validasi.append(validasi_accuracy[1]*100)

prediksi_train = model.predict_classes(training_feature)
prediksi_train_prob = model.predict(training_feature)
prediksi_test = model.predict_classes(testing_feature)
prediksi_test_prob = model.predict(testing_feature)
prediksi_vali = model.predict_classes(validasi_fitur)
prediksi_vali_prob = model.predict(validasi_fitur)

cm_value_training = confusion_matrix(training_labels,prediksi_train)
cm_value_training = np.array(cm_value_training)
validasi.append(cm_value_training)
print(cm_value_training)
print("\n")

cm_value_testing = confusion_matrix(testing_labels, prediksi_test)
cm_value_testing = np.array(cm_value_testing)
validasi.append(cm_value_testing)
print(cm_value_testing)
print("\n")

cm_value_validasi = confusion_matrix(validasi_label, prediksi_vali)
cm_value_validasi = np.array(cm_value_validasi)
validasi.append(cm_value_validasi)
print(cm_value_validasi)
print("\n")

Sen_Class = []
Spe_Class = []
Pre_Class = []
F1_Class = []
Err_Class = []
Acc_Class = []
FP_Class = []
FN_Class = []

for idx in range(len(cm_value_validasi)):
        TP = cm_value_testing[idx, idx]
        FP = np.sum(cm_value_testing[idx, :]) - TP
        FN = np.sum(cm_value_testing[:, idx]) - TP
        TN = np.sum(cm_value_testing) - (TP + FN + FP)
        
        Sen = TP / (TP + FN)
        Spe = TN / (TN + FP)
        Pre = TP / (TP + FP)
        F1 = (2 * Pre * Sen) / (Sen + Pre)
        Err = (FP + FN) / (FP + FN + TN + TP)
        Acc = (TP + TN) / (FP + FN + TN + TP)
        
        Sen_Class.append([Sen, idx])
        Spe_Class.append([Spe, idx])
        Pre_Class.append([Pre, idx])
        F1_Class.append([F1, idx])
        Err_Class.append([Err, idx])
        Acc_Class.append([Acc, idx])
        FP_Class.append([FP,idx])
        FN_Class.append([FN,idx])

sensi =[]
Sen_Class_numpy = np.array(Sen_Class)
print(Sen_Class_numpy)
avg_sen = np.mean(Sen_Class_numpy[:,0])
print("Rata-rata sensitivity: %.2f%%" % (avg_sen*100),"\n")
sensi.append(Sen_Class_numpy)
sensi.append(avg_sen)
validasi.append(sensi)

spesi =[]
Spe_Class_numpy = np.array(Spe_Class)
avg_spe = np.mean(Spe_Class_numpy[:,0])
print(Spe_Class_numpy)
print("Rata-rata specificity: %.2f%%" % (avg_spe*100),"\n")
spesi.append(Spe_Class_numpy)
spesi.append(avg_spe)
validasi.append(spesi)

presi =[]
Pre_Class_numpy = np.array(Pre_Class)
avg_pre = np.mean(Pre_Class_numpy[:,0])
print(Pre_Class_numpy)
print("Rata-rata precision: %.2f%%" % (avg_pre*100),"\n")
presi.append(Pre_Class_numpy)
presi.append(avg_pre)
validasi.append(presi)

f1scr =[]
F1_Class_numpy = np.array(F1_Class)
avg_F1 = np.mean(F1_Class_numpy[:,0])
print(F1_Class_numpy)
print("Rata-rata F1 score: %.2f%%" % (avg_F1*100),"\n")
f1scr.append(F1_Class_numpy)
f1scr.append(avg_F1)
validasi.append(f1scr)

errt =[]
Err_Class_numpy = np.array(Err_Class)
avg_err = np.mean(Err_Class_numpy[:,0])
print(Err_Class_numpy)
print("Rata-rata error rate: %.2f%%" % (avg_err*100),"\n")
errt.append(Err_Class_numpy)
errt.append(avg_err)
validasi.append(errt)

accu =[]
Acc_Class_numpy = np.array(Acc_Class)
avg_acc = np.mean(Acc_Class_numpy[:,0])
print(Acc_Class_numpy)
print("Rata-rata akurasi: %.2f%%" % (avg_acc*100),"\n")
accu.append(Acc_Class_numpy)
accu.append(avg_acc)
validasi.append(accu)

model.save('DNN.h5')
nam='Validasi'
pick = open(nam,'wb')
pickle.dump(validasi,pick)
pick.close()