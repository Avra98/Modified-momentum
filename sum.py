import torch 
from torch.optim.optimizer import Optimizer

class SUM(Optimizer):


    def __init__(self, params, lr=1e-1, momentum = 0.2,dampening =0, weight_decay=0,nesterov =False):

        defaults = dict(lr=lr, momentum = momentum, dampening=dampening,weight_decay=weight_decay,nesterov=nesterov)
        super(SUM, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(SUM, self).__setstate__(state)

    def step(self, iter, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()        

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            beta = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            ## Implements equation 0.3 in notes
            for p in group['params']:
                if p.grad is None:
                    continue
                if iter>0:
                    second_term = self.state[p]["pre_grad"].clone()

                d_p = p.grad.data
                first_term = d_p.clone()

                if iter > 0:                    
                    d_p = d_p.add(beta/1-beta,first_term.add(-1,second_term))
                if weight_decay != 0:
                    d_p.add_(p.data,weight_decay)
                # Apply learning rate  
                d_p.mul_(group['lr'])
                if beta != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(beta).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(beta).add_(1 - dampening,d_p)
                    if nesterov:
                        d_p = d_p.add(beta,buf)
                    else:
                        d_p = buf

                p.data.add_(-1,d_p)
        return loss 

    @torch.no_grad() 
    def store(self,zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["pre_grad"] = p.grad.data.clone()             
    
        if zero_grad: self.zero_grad()        

