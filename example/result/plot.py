import os 
import numpy as np
import matplotlib.pyplot as plt

root = './'
files = os.listdir(root)
for file in files:
    if ('adaptiveFalse.txt' in file) or ('sgd.txt' in file) and file[-3:] == 'txt': 
        SGDM_train,SGDM_test = [],[]
        SGD_train,SGD_test = [],[]
        #SUM_train,SUM_test = [],[]

        # SGD
        epoch_last = 0
        with open(root+file) as f: 
            for line in f.readlines():
                if "┃" not in line:
                    continue
                lst = line.split("┃")
                epoch = int(lst[1].replace(' ',''))
                if epoch < epoch_last:
                    SGD_train = []
                    SGD_test = []
                    
                SGD_train.append(float(lst[2].split("│")[1][:-3])/100.)
                SGD_test.append(float(lst[-2].split("│")[1][:-3])/100.)
                epoch_last=epoch
        best_sgd = round(np.max(SGD_test), 3)
        best_sgd_iter = np.argmax(SGD_test)

        # SGDM
        epoch_last = 0
        if 'adaptive' in file:
            file = file.replace("adaptiveFalse", "adaptiveTrue")
        else:
            file = file.replace("sgd", "sgdm")
        with open(root+file) as f:
            for line in f.readlines():
                if "┃" not in line:
                    continue
                lst = line.split("┃")
                epoch = int(lst[1].replace(' ',''))
                if epoch < epoch_last:
                    SGDM_train = []
                    SGDM_test = []

                SGDM_train.append(float(lst[2].split("│")[1][:-3])/100.)
                SGDM_test.append(float(lst[-2].split("│")[1][:-3])/100.)
                epoch_last=epoch
        best_sgdm = round(np.max(SGDM_test), 3)
        best_sgdm_iter = np.argmax(SGDM_test)

        beta = str(float(file.split('beta')[1][0])/10)
        lr   = str(float(file.split('lr')[1].split('beta')[0])/1e3)
        width = file.split('width')[1][0]
        depth = file.split('depth')[1].split('adaptive')[0]
        dataset = file.split('lr')[0]
        name = dataset +':'+ 'beta='+beta+',lr='+lr+','+'width='+width+',depth='+depth+'\n SGD:'+str(best_sgd)+', SGDM:'+str(best_sgdm)
        
        plt.figure(figsize=(8,5))
        plt.plot(SGD_train,'r',label='SGD_train_acc')
        plt.plot(SGD_test,'b', label='SGD_test_acc')
        plt.plot(SGDM_train,'y',label='SGDM_train_acc')
        plt.plot(SGDM_test,'c',label='SGDM_test_acc')
        #plt.plot(SUM_train,'g',label='SUM_train_acc')
        #plt.plot(SUM_test,'orange',label='SUM_test_acc')
        #plt.show()
        plt.legend()
        plt.title(name)
        plt.tick_params(labelright=True, left=True, right=True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        #plt.ylim((0.9,1.01))
        plt.savefig('./imgs/'+name+'.png')
 
# with open('./result/cifar100_sgd_beta08.txt') as f:
#     for line in f.readlines():
#         if "┃" not in line:
#             continue
#         lst = line.split("┃")
#         #print(lst)
#         #epoch = int(lst[1]) - 1
#         SGD_train.append(float(lst[2].split("│")[1][:-3])/100.)
#         SGD_test.append(float(lst[-2].split("│")[1][:-3])/100.)


# with open('./result/cifar100_sum_beta08.txt') as f:
#     for line in f.readlines():
#         if "┃" not in line:
#             continue
#         lst = line.split("┃")
#         #print(lst)
#         #epoch = int(lst[1]) - 1
#         SUM_train.append(float(lst[2].split("│")[1][:-3])/100.)
#         SUM_test.append(float(lst[-2].split("│")[1][:-3])/100.)
        
