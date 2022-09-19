import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

root = './'
lr_select = 0.1
data = 'cifar100'
seed_select = '42'

files = os.listdir(root)
SGD_train,SGD_test = {},{}

for file in files:
    if file[-3:] == 'txt': 
        
        beta = str(float(file.split('beta')[1][0])/10)
        lr   = str(float(file.split('lr')[1].split('beta')[0])/1e3)
        model= file.split('model')[1].split('seed')[0]
        seed = file.split('seed')[1].split('scheduler')[0]
        dataset = file.split('lr')[0]
        scheduler = file.split('scheduler')
       
        if len(scheduler) == 1:
            scheduler = 'False'
        else:
            scheduler = scheduler[1].split('.')[0]
              
        if abs(float(lr)/(1-float(beta)) - lr_select)>1e-3 or data != dataset or seed != seed_select: 
            continue
        
        if model == 'wide': model = 'wideresenet'
        if dataset == 'fashion': dataset = 'fashionmnist'

        key = dataset +':'+'beta='+beta+',lr='+lr+',('+model+')'
       
        if model not in SGD_test:
            SGD_test[model] = {}
            SGD_train[model] = {}

        SGD_train[model][key] = []
        SGD_test[model][key] = []

        epoch_last = 0
        with open(root+file) as f: 
            for line in f.readlines():
                if "┃" not in line:
                    continue
                lst = line.split("┃")
                epoch = int(lst[1].replace(' ',''))
                if epoch < epoch_last:
                    SGD_train[model][key] = []
                    SGD_test[model][key] = []
                    
                SGD_train[model][key].append(float(lst[2].split("│")[1][:-3])/100.)
                SGD_test[model][key].append(float(lst[-2].split("│")[1][:-3])/100.)
                #SGD_train[model][key].append(float(lst[2].split("│")[0][:-3]))
                #SGD_test[model][key].append(float(lst[-2].split("│")[0][:-3]))

                epoch_last=epoch

colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3']
for model in SGD_train:       
    plt.figure(figsize=(6,10))

    best_sgd = 0
    for i, key in enumerate(sorted(SGD_train[model].keys())):

        best_sgd_i = round(np.max(SGD_test[model][key]), 3)
        if best_sgd_i > best_sgd:
            best_sgd = best_sgd_i
            k = key

        label = key.split(':')[1].split('(')[0]
        
        train = pd.Series(SGD_train[model][key])
        train_mean = train.rolling(5).mean().values[4:]
        train_std  = train.rolling(5).std().values[4:]
        plt.plot(train_mean, c = colors[i], label=label+'train')
        plt.fill_between(np.arange(len(train_mean)), (train_mean-train_std), (train_mean+train_std), color = colors[i], alpha=.1)

        test = pd.Series(SGD_test[model][key])
        test_mean = test.rolling(5).mean().values[4:]
        test_std  = test.rolling(5).std().values[4:]
        plt.plot(test_mean, c = colors[i], label=label+'test', linestyle='dashed')
        plt.fill_between(np.arange(len(test_mean)), (test_mean-test_std), (test_mean + test_std), color = colors[i], alpha=.1)

    #print(k, best_sgd)

    plt.legend()
    plt.title(model)
    plt.tick_params(labelright=True, left=True, right=True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.ylim([0.80,1.0])
    plt.savefig('./imgs/'+data+'_lr'+str(int(lr_select*1e3))+'_'+model+'.png')
