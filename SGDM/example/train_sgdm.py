import argparse
import string
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from data.mnist import MNIST
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sgdm import SGDM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="cifar",type=str,help="Specify dataset")
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=1e-1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.2, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0000, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--path",default="./result/cifar10.txt",type=str,help="Path to store the results")
    parser.add_argument("--increment",default=1,type=int,help="Add incremental terms")
    #parser.add_argument("dampening", default=0.0, type=float,help="dampening")
    #parser.add_argument("nesterov", default=False, type=bool,help="nesterov")
    args = parser.parse_args()
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dataset.startswith("cifar"):
        dataset = Cifar(args.batch_size, args.threads)
    else:
        dataset = MNIST(args.batch_size, args.threads)
    log = Log(log_each=10,path=args.path)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=1, labels=10).to(device)

    #base_optimizer = torch.optim.SGD
    optimizer = SGDM(model.parameters(), lr=args.learning_rate, momentum=args.momentum, dampening=0, weight_decay=args.weight_decay,nesterov = False)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        iter = 0
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # get the first term \grad E_{n}(x(tn)+\delta x1) and store it in self.state[p]["pres_grad"] (look at sgdm.py))
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            
            optimizer.first_step(zero_grad=True,increment=args.increment)

            disable_running_stats(model)
            loss = smooth_crossentropy(model(inputs),targets, smoothing=args.label_smoothing)
            loss.mean().backward()

            optimizer.second_step(zero_grad=False,mode ="first")

            ### make the original update
            optimizer.step(iter)


            ### store the second term to be used for next iteration. Store it in self.state[p]["pre_grad"] (look at sgdm.py))
            enable_running_stats(model)
            loss = smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True,increment=args.increment)
 
            disable_running_stats(model)
            loss = smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.second_step(zero_grad=True,mode ="second")
            iter=iter+1

            
            
            

            with torch.no_grad():
                predictions1 = model(inputs)
                correct = torch.argmax(predictions1.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
        optimizer.zero_grad()

        model.eval()
        log.eval(len_dataset=len(dataset.test))
    
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions2 = model(inputs)
                loss = smooth_crossentropy(predictions2, targets)
                correct = torch.argmax(predictions2, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()