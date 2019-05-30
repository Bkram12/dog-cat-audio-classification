# -*- coding: utf-8 -*-
"""
Created on Sat May 25 12:30:52 2019

@author: mark
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
import librosa
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


#%% view some data
DIR = './cats_dogs'
files = os.listdir(f'{DIR}')
split_data= pd.read_csv('./train_test_split.csv')
filt_len = int(0.4*16000)
plt.figure()
fs=16000

def load_files(files):
    data = pd.DataFrame()
    for file in files:
        data = wavfile.read(f'./cats_dogs/{file}')
    return(data)

for i in range(3):

    plt.subplot(2,3,i+1)
    file = split_data['test_cat'][i]
    wav_file= wavfile.read(f'./cats_dogs/{file}')[1]
    wav_mean = pd.DataFrame(abs(wav_file)).rolling(window = filt_len, min_periods=1, center=True).mean()

    plt.plot(wav_file)
    plt.plot(wav_mean)
    plt.title(file)

for i in range(3,6):

    plt.subplot(2,3,i+1)
    file = split_data['test_dog'][i-3]
    wav_file= wavfile.read(f'./cats_dogs/{file}')[1]
    wav_mean = pd.DataFrame(abs(wav_file)).rolling(window = filt_len, min_periods=1, center=True).mean()
    plt.plot(wav_file)
    plt.plot(wav_mean)
    plt.title(file)
#%% cleanup data

def mask_low_vol(wav_file, threshold):
    mask = []
    wav_file = pd.Series(wav_file).abs() # load file
    filt_len = int(0.4*16000)
#    filt_len = int(len(wav_file)/20)
    wav_mean = wav_file.rolling(window = filt_len, min_periods=1, center=True).mean()
    for mean in wav_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def clean_data(files):

    threshold=250
    for file in tqdm(files):
        wav_file = wavfile.read(f'./cats_dogs/{file}')[1]
        mask = mask_low_vol(wav_file, threshold)
        wav_file = wav_file[mask]
        if len(wav_file)>0:
            while (len(wav_file)<LEN*fs):
                wav_file = np.append(wav_file,wav_file)
            wav_file=wav_file[0:LEN*fs]
            wavfile.write(filename=f'./removed_noise/{file}',rate=fs, data=wav_file)
LEN = 1
files = os.listdir(f'./cats_dogs')
clean_data(files)

#%% Look at clean files
plt.figure()

for i in range(3):

    plt.subplot(2,3,i+1)
    file = split_data['test_cat'][i]
    wav_file= wavfile.read(f'./removed_noise/{file}')[1]
    wav_mean = pd.DataFrame(abs(wav_file)).rolling(window = filt_len, min_periods=1, center=True).mean()
    plt.plot(wav_file)
    plt.plot(wav_mean)

    plt.title(file)

for i in range(3,6):

    plt.subplot(2,3,i+1)
    file = split_data['test_dog'][i-3]
    wav_file= wavfile.read(f'./removed_noise/{file}')[1]
    wav_mean = pd.DataFrame(abs(wav_file)).rolling(window = filt_len, min_periods=1, center=True).mean()
    plt.plot(wav_file)
    plt.plot(wav_mean)
    plt.title(file)

#%% create fft from data & save files

def create_fft(file):

    fs, wav_file = wavfile.read(f'./removed_noise/{file}')
    try:

        fft = np.fft.rfft(wav_file)
        np.save(f'./fft/fft-{file[:-4]}.npy',fft)
#
    except:
        print(f'skipping{file}')


files = os.listdir(f'./removed_noise')
for file in tqdm(files):
    create_fft(file)


#%% create train & test data

def get_data(files):
    data = []
    for file in files:
#        data.append(abs(np.load(f'./fft_10sec/fft-{file[:-4]}.npy')))
        if file not in split_data['odd'].values:
            data.append(abs(np.load(f'./fft_10sec/fft-{file[:-4]}.npy')))
    return (data)

X_train = []
X_validate = []
y_train = []
y_validate = []



X_train = get_data(split_data['train_cat'].dropna())
y_train = np.ones((1,len(X_train)))
n_samples=y_train.shape[1]
temp = get_data(split_data['train_dog'].dropna())
X_train = X_train+temp
y_train = np.append(y_train, (np.zeros((1,len(X_train)-n_samples))))


X_validate = get_data(split_data['test_cat'].dropna())
y_validate = np.ones((1,len(split_data['test_cat'].dropna())))
n_samples=y_validate.shape[1]
X_validate = X_validate + (get_data(split_data['test_dog'].dropna()))
y_validate = np.append(y_validate, (np.zeros((1,len(X_validate)-n_samples))))

X_train = np.array(X_train)
X_validate = np.array(X_validate)

X_train = StandardScaler().fit_transform(X_train)
X_validate = StandardScaler().fit_transform(X_validate)

X_train,_,y_train,_= train_test_split(X_train,y_train,test_size=0.001)

pca = PCA(n_components = 9)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_validate_pca = pca.transform(X_validate)

#%% create model
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

models = [
            ['KNClassifier', KNeighborsClassifier(n_neighbors=3)],
            ['DTClassifier', DecisionTreeClassifier(max_depth=20)],
            ['RFClassifier', RandomForestClassifier(max_depth=20, n_estimators=20)],
            ['LogReg', LogisticRegression(solver='lbfgs')]
            ['GPClassifier',GaussianProcessClassifier(1.0 * RBF(1.0))],
            ['ABClassifier', AdaBoostClassifier()],
            ['SVMlin', SVC(kernel='linear', gamma = 'scale', probability=True)],
            ['SVMpoly', SVC(kernel='poly', gamma = 'scale', probability=True)],
            ['SVMrbf', SVC(kernel='rbf', gamma = 'scale', probability=True)],
            ['SVMsig', SVC(kernel='sigmoid', gamma = 'scale', probability=True)],
            ['MLPClassifier(10,10,10)', MLPClassifier(hidden_layer_sizes=(20,20,20), activation='relu', solver='adam',max_iter=1000 ,early_stopping=True)],
            ['MLPClassifier(100,100,100)', MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu', solver='adam',max_iter=1000 ,early_stopping=True)],
           ]

model_data=[]
kf = KFold(n_splits=10)

for index, (name, curr_model) in enumerate(models):
    print(name)
    curr_model_data={}
    curr_model_data['Name']=name
    score=[]
    for train_index, test_index in kf.split(X_train_pca):
        curr_model.fit(X_train_pca[train_index], y_train[train_index])
        y_pred = curr_model.predict(X_train_pca[test_index])
        score.append(f1_score(y_train[test_index],y_pred))
    curr_model.fit(X_train_pca, y_train)
    y_pred = curr_model.predict(X_validate_pca)
    val_score= f1_score(y_validate,y_pred)
    model_data.append(([name,np.mean(score),val_score]))

model_data = pd.DataFrame(model_data)
model_data.columns = ['model','f1_testdata','f1_valdata']

print(model_data)

#plt.figure()
plt.scatter(model_data['f1_testdata'],model_data['f1_valdata'])
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0,1],c='black')
plt.xlabel('test')
plt.ylabel('validate')
plt.grid()
#%%

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu', solver='adam',max_iter=1000 ,early_stopping=True)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_validate_pca)
y_tp = clf.predict(X_train_pca)


print('train data')
print(confusion_matrix(y_train,y_tp))
print('validation')
print(confusion_matrix(y_validate,y_pred))
print(classification_report(y_validate, y_pred))

#%% look at errors

y_error=y_validate-y_pred
filenames=pd.concat([split_data['test_cat'].dropna(),split_data['test_dog'].dropna()],ignore_index=True)

for i in range(len(y_error)):
    if y_error[i]!=0:
        print(filenames[i])

# Something I've added as a github test
