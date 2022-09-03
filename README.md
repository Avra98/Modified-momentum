# modified-momentum
This is an modified version of the SAM implementation. Under construction. 

To run the SGDM, the implementation in eq 0.3 do
python example/train_sgdm.py --dataset cifar10 --depth 10 --width 4 --label_smoothing 0.1 -lr 1e-1 -beta 0.8 --epochs 500 

To run the SUM, the implementation in eq 0.83 do
python example/train_sum.py

To run the SGD, the implementation in eq 0.1 do
python example/train_sgd.py
or
python example/train_sgdm.py -apt --dataset cifar10 --depth 10 --width 4 --label_smoothing 0.1 -lr 1e-1 -beta 0.8 --epochs 500 
