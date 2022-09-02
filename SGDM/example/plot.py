import matplotlib.pyplot as plt
SGDM_train,SGDM_test = [],[]
SGD_train,SGD_test = [],[]
with open('./result/cifar10_beta04_width1_sgdm.txt') as f:
    for line in f.readlines():
        if "┃" not in line:
            continue
        lst = line.split("┃")
        #print(lst)
        #epoch = int(lst[1]) - 1
        SGDM_train.append(float(lst[2].split("│")[1][:-3])/100.)
        SGDM_test.append(float(lst[-2].split("│")[1][:-3])/100.)

with open('./result/cifar10_beta04_width1_noincre.txt') as f:
    for line in f.readlines():
        if "┃" not in line:
            continue
        lst = line.split("┃")
        #print(lst)
        #epoch = int(lst[1]) - 1
        SGD_train.append(float(lst[2].split("│")[1][:-3])/100.)
        SGD_test.append(float(lst[-2].split("│")[1][:-3])/100.)

plt.plot(SGD_train,'r',label='noincre_train_acc')
plt.plot(SGD_test,'b',label='noincre_test_acc')
plt.plot(SGDM_train,'y',label='SGDM_train_acc')
plt.plot(SGDM_test,'c',label='SGDM_test_acc')
plt.show()
plt.legend()
plt.title('cifar10 beta=0.4 width=1')
plt.savefig('./result/cifar10_beta04_width1_compare_incre.jpg')
        