import sys; sys.path.append("..")

import argparse
import torch, torchvision

from torch.optim import SGD
from data.cifar import Cifar10, Cifar100, FashionMNIST, MNIST, Cifar10Sub
from utility.log import Log
from model.resnet import *
from model.resnetnbn import *
from model.densenet import *
from model.small import *
from model.vgg import *
from utility.sam import SAM
from model.wide_res_net import WideResNet
from utility.initialize import initialize
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='resnet18', type=str, help="select optimizer")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=300, type=int, help="Total number of epochs.")
    parser.add_argument("--learning_rate", '-lr', default=1e-1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", '-beta', default=0.8, type=float, help="SGD Momentum.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset name")
    parser.add_argument("--threads", default=4, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0000, type=float, help="L2 weight decay.")
    parser.add_argument("--seed", default=42, type=int, help="L2 weight decay.")
    parser.add_argument("--patience", default=250, type=int, help="patience for scheduler.")
    parser.add_argument("--scheduler", default='stepLR', type=str, help="select scheduler.")
    parser.add_argument("--multigpu", "-m", action='store_true', help="whether using multi-gpus.")
    parser.add_argument("--noise_level", '-nl', type=float, default=0.0, help="noise level for PGD")
    parser.add_argument("--optimizer", type=str, default='SGD', help="SGD/SAM")

    args = parser.parse_args()

    initialize(args, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = 10
    size = (32, 32)
    if args.dataset.lower() == 'cifar10':
        dataset = Cifar10(args.batch_size, args.threads, size)
    elif args.dataset.lower() == 'cifar100':
        dataset = Cifar100(args.batch_size, args.threads, size)
        labels = 100
    elif args.dataset.lower() == 'fashionmnist':
        dataset = FashionMNIST(args.batch_size, args.threads, size)
    elif args.dataset.lower() == 'cifar10sub':
        dataset = Cifar10Sub(args.batch_size, args.threads, size)
    else:
        dataset = MNIST(args.batch_size, args.threads, size)
        
    
    if args.model.lower() == 'resnet18':
        model = ResNet18(num_classes=labels).to(device)
    elif args.model.lower() == 'resnet34':
        model = ResNet34(num_classes=labels).to(device)
    elif args.model.lower() == 'resnet50':
        model = ResNet50(num_classes=labels).to(device)
    elif args.model.lower() == 'densenet121':
        model = DenseNet121(num_classes=labels).to(device)
    elif args.model.lower() == 'wide':
        model = WideResNet(depth=16, width_factor=8, dropout=0.0, 
                    in_channels=3, labels=labels).to(device)
    elif args.model.lower() == 'resnet18nbn':
        model = ResNet18nbn(num_classes=labels).to(device)
    elif args.model.lower() == 'resnet50nbn':
        model = ResNet50nbn(num_classes=labels).to(device)
    elif args.model.lower() == 'vgg13':
        model = VGG('VGG13',num_classes=labels).to(device) 
    elif args.model.lower() == 'vgg16':
        model = VGG('VGG16',num_classes=labels).to(device)    
    else:
        model = smallnet(num_classes=labels).to(device)
    
    if args.multigpu:
        model = torch.nn.DataParallel(model, device_ids=[0,1])

    log = Log(log_each=10, file_name= args.dataset+'lr'+str(int(1e3*args.learning_rate))
                                          +'beta'+str(int(10*args.momentum))
                                          +'model'+str(args.model)
                                          +'seed'+str(args.seed)
                                          +'scheduler'+args.scheduler
                                          +'patience'+str(args.patience)
                                          +'nl'+str(int(1e4*args.noise_level))
                                          +'optimizer'+args.optimizer)
    #if 'nbn' not in args.model.lower():
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    #else:
    #    criterion = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=1.0/labels)

    if args.optimizer.lower() == 'sgd':
        optimizer = SGD(model.parameters(),lr=args.learning_rate, momentum=args.momentum, 
                        weight_decay=args.weight_decay, nesterov=False)
    else:
        optimizer = SAM(model.parameters(), SGD, lr=args.learning_rate, momentum=args.momentum)
    
    if args.scheduler.lower() == 'steplr':
        scheduler = StepLR(optimizer, step_size = args.patience, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience = args.patience, factor=0.1)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))
        for batch in dataset.train:

            def closure():
                loss = criterion(model(inputs), targets)
                loss.mean().backward()
                return loss

            if args.noise_level > 0:
                noise_list = []
                for param in model.parameters():
                    noise = torch.randn(param.data.size()).to(device)*args.noise_level
                    param.data += noise
                    noise_list.append(noise)

            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.mean().backward()

            if args.optimizer.lower() == 'sam':
                optimizer.step(closure)
            else:
                optimizer.step()

            if args.noise_level > 0:
                for param in model.parameters():
                    param.data -= noise_list[0].to(device)
                    noise_list = noise_list[1:]

            correct = torch.argmax(predictions.data, 1) == targets
            log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]['lr'])
        
            def closure():
                loss = criterion(output, model(input))
                loss.backward()
                return loss

           

        model.eval()
        log.eval(len_dataset=len(dataset.test))
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu(), optimizer.param_groups[0]['lr'])
        if args.scheduler:
            scheduler.step(loss.mean())
           
    log.flush()
