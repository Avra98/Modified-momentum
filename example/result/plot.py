import os 
import numpy as np
import matplotlib.pyplot as plt

root = './logs/'
files = os.listdir(root)
SGD_train,SGD_test = {},{}

for file in files:
    if file[-3:] == 'txt': 
        
        beta = str(float(file.split('beta')[1][0])/10)
        lr   = str(float(file.split('lr')[1].split('beta')[0])/1e3)
        model= file.split('model')[1].split('.')[0]
        dataset = file.split('lr')[0]

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


#colors = ['#1b9e77','#d95f02','#7570b3','#e7298a']
for model in SGD_train:       
    plt.figure(figsize=(8,5))

    best_sgd = 0
    for i, key in enumerate(SGD_train[model]):

        best_sgd_i = round(np.max(SGD_test[model][key]), 3)
        if best_sgd_i > best_sgd:
            best_sgd = best_sgd_i
            k = key

        label = key.split(':')[1].split('(')[0]
        plt.plot(SGD_train[model][key],label=label+'train')
        plt.plot(SGD_test[model][key] ,label=label+'test', linestyle='dashed')
    #print(k, best_sgd)

    plt.legend()
    plt.title(model)
    plt.tick_params(labelright=True, left=True, right=True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.ylim([0.90,0.96])
    plt.savefig('./imgs/'+model+'.png')