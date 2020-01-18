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
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pickle

encod = 126
hidden = 100
layout = 5

dam = Adam(lr=0.0001)
fitur = np.load('data/feature_encoded.npy')
label = np.load('../../data/label.npy')
label = to_categorical(label,5)

training_feature, testing_feature, training_label, testing_label = train_test_split(fitur, label, test_size = 0.1, stratify=(label))

model = Sequential()
model.add(Dense(units = hidden, input_shape=(encod, ), use_bias=False, activation='relu'))
model.add(Dense(units = hidden, use_bias=False, activation='relu'))
model.add(Dense(units = hidden, use_bias=False, activation='relu'))
model.add(Dense(units = hidden, use_bias=False, activation='relu'))
model.add(Dense(units = hidden, use_bias=False, activation='relu'))
model.add(Dense(units = layout , use_bias=False, activation='softmax'))
model.compile(optimizer=dam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training_feature, training_label, epochs=200, batch_size=48, shuffle=True, validation_data=(testing_feature, testing_label))
model.summary()
print("\n")

validasi =[] 

rt_loss_train = np.average(model.history.history['loss'])
validasi.append(rt_loss_train)
rt_acc_train = np.average(model.history.history['acc'])
validasi.append(rt_acc_train)
rt_loss_test = np.average(model.history.history['val_loss'])
validasi.append(rt_loss_test)
rt_acc_test = np.average(model.history.history['val_acc'])
validasi.append(rt_acc_test)
max_acc_train = np.max(model.history.history['acc'])
max_acc_test = np.max(model.history.history['val_acc'])

print("rata-rata loss training: ",  rt_loss_train)
print("rata-rata acc training: %.2f%%" % (rt_acc_train*100))
print("rata-rata loss testing: ",  rt_loss_test)
print("rata-rata acc testing: %.2f%%" % (rt_acc_test*100),"\n")
print("maks acc training: %.2f%%" % (max_acc_train*100))
print("mask acc testing: %.2f%%" % (max_acc_test*100),"\n")

plt.figure()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.ylim((0,1))
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.savefig('data/loss.png')
plt.show()

plt.figure()
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.ylim((0,1))
plt.xlabel('epoch')
plt.legend(['train','test'], loc='lower right')
plt.savefig('data/acc.png')
plt.show()

testing_accuracy = model.evaluate(testing_feature, testing_label)
print('Loss validasi {0}'.format(str(testing_accuracy[0])))
print('Akurasi validasi {0}%'.format(str(testing_accuracy[1]*100),"\n"))

prediksi_train = model.predict_classes(training_feature)
prediksi_train_prob = model.predict(training_feature)
training_label_non_kategori = [np.argmax(t) for t in training_label]

prediksi_test = model.predict_classes(testing_feature)
prediksi_test_prop = model.predict(testing_feature)
testing_label_non_kategori = [np.argmax(t) for t in testing_label]

cm_value_training = confusion_matrix(training_label_non_kategori,prediksi_train)
cm_value_training = np.array(cm_value_training)
validasi.append(cm_value_training)
print(cm_value_training, '\n')

cm_value_testing = confusion_matrix(testing_label_non_kategori, prediksi_test)
cm_value_testing = np.array(cm_value_testing)
validasi.append(cm_value_testing)
print(cm_value_testing, '\n')

plt.figure()
plt.imshow(cm_value_testing, interpolation='nearest', cmap=plt.cm.Blues)
plt.tight_layout()
plt.colorbar()
tick_marks = np.arange(layout)
plt.xticks(tick_marks, range(layout))
plt.yticks(tick_marks, range(layout))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('data/Confussion Matrix plot.png')
plt.show()

Sen_Class = []
Spe_Class = []
Pre_Class = []
F1_Class = []
Err_Class = []
Acc_Class = []
TP_Class = []
FP_Class = []
TN_Class = []
FN_Class = []

for idx in range(len(cm_value_testing)):
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
        TP_Class.append([TP,idx])
        FP_Class.append([FP,idx])
        TN_Class.append([TN,idx])
        FN_Class.append([FN,idx])

TP_Class = np.array(TP_Class)
validasi.append(TP_Class)
FP_Class = np.array(FP_Class)
validasi.append(FP_Class)
TN_Class = np.array(TN_Class)
validasi.append(TN_Class)
FN_Class = np.array(FN_Class)
validasi.append(FN_Class)
 
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

model.save('data/DNN.h5')
np.save('data/fitur_test.npy', testing_feature)
np.save('data/label_test.npy', testing_label)
nam='data/Validasi'
pick = open(nam,'wb')
pickle.dump(validasi,pick)
pick.close()

with open('data/hasil.txt', 'w') as tes:
    tes.write('rata-rata loss training {0} \n'.format(str(rt_loss_train)))
    tes.write('rata-rata akurasi training {0} \n'.format(str(rt_acc_train)))
    tes.write('rata-rata loss testing {0} \n'.format(str(rt_loss_test)))
    tes.write('rata-rata akurasi testing {0} \n\n'.format(str(rt_acc_test)))
    tes.write('Loss validasi {0} \n'.format(str(testing_accuracy[0])))
    tes.write('Akurasi validasi {0}% \n\n'.format(str(testing_accuracy[1]*100),"\n"))
    tes.write('Confussion Matrix training \n{0} \n\n'.format(cm_value_training))
    tes.write('Confussion Matrix testing \n{0} \n\n'.format(cm_value_testing))
    tes.write('TP \n{0} \n'.format(TP_Class))
    tes.write('FP \n{0} \n'.format(FP_Class))
    tes.write('TN \n{0} \n'.format(TN_Class))
    tes.write('FN \n{0} \n\n'.format(FP_Class))
    tes.write('Sensitivas \n{0} \n{1}\n\n'.format(Sen_Class_numpy,avg_sen))
    tes.write('Spesifisitas \n{0} \n{1}\n\n'.format(Spe_Class_numpy,avg_spe))
    tes.write('Presisi \n{0} \n{1}\n\n'.format(Pre_Class_numpy,avg_pre))
    tes.write('F1 \n{0} \n{1}\n\n'.format(F1_Class_numpy,avg_F1))
    tes.write('Error \n{0} \n{1}\n\n'.format(Err_Class_numpy,avg_err))
    tes.write('Akurasi \n{0} \n{1}\n\n'.format(Acc_Class_numpy,avg_acc))