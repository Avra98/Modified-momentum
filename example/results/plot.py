import matplotlib.pyplot as plt
SGDM_train,SGDM_test = [],[]
SGD_train,SGD_test = [],[]
SUM_train,SUM_test = [],[]


with open('./result/cifar100_sgdm_beta08.txt') as f:
    for line in f.readlines():
        if "┃" not in line:
            continue
        lst = line.split("┃")
        #print(lst)
        #epoch = int(lst[1]) - 1
        SGDM_train.append(float(lst[2].split("│")[1][:-3])/100.)
        SGDM_test.append(float(lst[-2].split("│")[1][:-3])/100.)

with open('./result/cifar100_sgd_beta08.txt') as f:
    for line in f.readlines():
        if "┃" not in line:
            continue
        lst = line.split("┃")
        #print(lst)
        #epoch = int(lst[1]) - 1
        SGD_train.append(float(lst[2].split("│")[1][:-3])/100.)
        SGD_test.append(float(lst[-2].split("│")[1][:-3])/100.)


with open('./result/cifar100_sum_beta08.txt') as f:
    for line in f.readlines():
        if "┃" not in line:
            continue
        lst = line.split("┃")
        #print(lst)
        #epoch = int(lst[1]) - 1
        SUM_train.append(float(lst[2].split("│")[1][:-3])/100.)
        SUM_test.append(float(lst[-2].split("│")[1][:-3])/100.)


plt.plot(SGD_train,'r',label='SGD_train_acc')
plt.plot(SGD_test,'b',label='SGD_test_acc')
plt.plot(SGDM_train,'y',label='SGDM_train_acc')
plt.plot(SGDM_test,'c',label='SGDM_test_acc')
plt.plot(SUM_train,'g',label='SUM_train_acc')
plt.plot(SUM_test,'orange',label='SUM_test_acc')
plt.show()
plt.legend()
plt.title('beta=0.8,lr=0.1 CIFAR-100 claasification')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('./result/cifar100_beta08.jpg')
        