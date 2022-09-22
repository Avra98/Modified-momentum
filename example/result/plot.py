import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

root = './vgg19_2560/'
lr_select = 0.1
data = 'cifar10'
seed_select = ['42','100','1000']
winSize = 5
break_plot = True 

files = os.listdir(root)
SGD_train,SGD_test = {},{}

for file in files:
    if file[-3:] == 'txt': 
        
        beta = str(float(file.split('beta')[1][0])/10)
        lr   = str(float(file.split('lr')[1].split('beta')[0])/1e3)
        model= file.split('model')[1].split('seed')[0]
        if "implicit" not in file:
            seed = file.split('seed')[1].split('scheduler')[0]
        else:
            seed = file.split('seed')[1].split('implicit')[0]

        dataset = file.split('lr')[0]
        scheduler = file.split('scheduler')
       
        if len(scheduler) == 1:
            scheduler = 'False'
        else:
            scheduler = scheduler[1].split('.')[0]
              
        if abs(float(lr)/(1-float(beta)) - lr_select)>1e-3 or data != dataset or seed not in seed_select: 
            continue
        
        if model == 'wide': model = 'wideresenet'
        if dataset == 'fashion': dataset = 'fashionmnist'
        if 'nbn' in model: model = model[:-3]

        key = dataset +':'+'beta='+beta+',lr='+lr+',('+model+')'
        if "implicit" in file:
            key = dataset +':'+'beta='+beta+',lr='+lr+",implicit"+',('+model+')'
       
        if model not in SGD_test:
            SGD_test[model] = {}
            SGD_train[model] = {}

        if key not in SGD_train[model]:
            SGD_train[model][key] = {}
            SGD_test[model][key] = {}

        SGD_train[model][key][seed] = []
        SGD_test[model][key][seed] = []           

        epoch_last = 0
        with open(root+file) as f: 
            for line in f.readlines():
                if "┃" not in line:
                    continue
                lst = line.split("┃")
                epoch = int(lst[1].replace(' ',''))
                if epoch < epoch_last:
                    SGD_train[model][key][seed] = []
                    SGD_test[model][key][seed] = []
                
                SGD_train[model][key][seed].append(float(lst[2].split("│")[1][:-3])/100.)
                SGD_test[model][key][seed].append(float(lst[-2].split("│")[1][:-3])/100.)
                #SGD_train[model][key][seed].append(float(lst[2].split("│")[0][:-3]))
                #SGD_test[model][key][seed].append(float(lst[-2].split("│")[0][:-3]))

                epoch_last=epoch

colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
if break_plot == False:
    for model in SGD_train:       
        plt.figure(figsize=(8,6))

        for i, key in enumerate(sorted(SGD_train[model].keys())):
            label = key.split(':')[1].split('(')[0]
            
            train = []
            for seed in SGD_train[model][key]:
                # train = pd.Series(SGD_train[model][key])
                # train_mean = train.rolling(5).mean().values[4:]
                # train_std  = train.rolling(5).std().values[4:]
                train.append(SGD_train[model][key][seed])
            train = np.array(train)
            train_mean = []
            train_std = []
            for j in range(0, train.shape[1]-winSize+1):
                window = train[:,j:j+winSize]
                train_mean.append(np.mean(window))
                train_std.append(np.std(window))
            train_mean, train_std = np.array(train_mean), np.array(train_std)
            plt.plot(train_mean, c = colors[i], label=label+'train')
            plt.fill_between(np.arange(len(train_mean)), (train_mean-train_std), (train_mean+train_std), color = colors[i], alpha=.1)

            test = []
            for seed in SGD_test[model][key]:
                test.append(SGD_test[model][key][seed])
            test = np.array(test)
            test_mean = []
            test_std = []
            for j in range(0, test.shape[1]-winSize+1):
                window = test[:,j:j+winSize]
                test_mean.append(np.mean(window))
                test_std.append(np.std(window))
            test_mean, test_std = np.array(test_mean), np.array(test_std)
            plt.plot(test_mean, c = colors[i], label=label+'test', linestyle='dashed')
            plt.fill_between(np.arange(len(test_mean)), (test_mean-test_std), (test_mean + test_std), color = colors[i], alpha=.1)

        plt.legend()
        plt.title(model)
        plt.tick_params(labelright=True, left=True, right=True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('./imgs/'+data+'_lr'+str(int(lr_select*1e3))+'_'+model+'.png')

else:

    for model in SGD_train:       
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6),gridspec_kw={'height_ratios': [3, 1]})
        for i, key in enumerate(sorted(SGD_train[model].keys())):
            label = key.split(':')[1].split('(')[0]
            
            train = []
            for seed in SGD_train[model][key]:
                train.append(SGD_train[model][key][seed])
            train = np.array(train)
            train_mean = []
            train_std = []
            for j in range(0, train.shape[1]-winSize+1):
                window = train[:,j:j+winSize]
                train_mean.append(np.mean(window))
                train_std.append(np.std(window))
            train_mean, train_std = np.array(train_mean), np.array(train_std)

            test = []
            for seed in SGD_test[model][key]:
                test.append(SGD_test[model][key][seed])
            test = np.array(test)
            test_mean = []
            test_std = []
            for j in range(0, test.shape[1]-winSize+1):
                window = test[:,j:j+winSize]
                test_mean.append(np.mean(window))
                test_std.append(np.std(window))
            test_mean, test_std = np.array(test_mean), np.array(test_std)

            ax1.plot(test_mean, c = colors[i], label=label+'test', linestyle='dashed')
            ax1.fill_between(np.arange(len(test_mean)), (test_mean-test_std), (test_mean + test_std), color = colors[i], alpha=.1)
            ax1.plot(train_mean, c = colors[i], label=label+'train')
            ax1.fill_between(np.arange(len(train_mean)), (train_mean-train_std), (train_mean+train_std), color = colors[i], alpha=.1)

            ax2.plot(test_mean, c = colors[i], label=label+'test', linestyle='dashed')
            ax2.fill_between(np.arange(len(test_mean)), (test_mean-test_std), (test_mean + test_std), color = colors[i], alpha=.1)
            ax2.plot(train_mean, c = colors[i], label=label+'train')
            ax2.fill_between(np.arange(len(train_mean)), (train_mean-train_std), (train_mean+train_std), color = colors[i], alpha=.1)

            ax1.set_ylim(.75, 1.) 
            ax2.set_ylim(.35, .6)

            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            d = .8
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                          linestyle='none', color='black', mec='black', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        ax1.legend()
        ax1.title.set_text(model)
        ax1.tick_params(labelright=True, left=True, right=True, bottom=False)
        ax2.tick_params(labelright=True, left=True, right=True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('./imgs/'+data+'_lr'+str(int(lr_select*1e3))+'_'+model+'.png')



