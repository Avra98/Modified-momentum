import torch 
from torch.optim.optimizer import Optimizer

class SUM_exact(Optimizer):


    def __init__(self, params, t = 0.001, lr=1e-1, momentum = 0.2,dampening =0, weight_decay=0,nesterov =False):

        defaults = dict(lr=lr, momentum = momentum, t=t, dampening=dampening,weight_decay=weight_decay,nesterov=nesterov)
        super(SUM_exact, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(SUM_exact, self).__setstate__(state)

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
            ## Implements the following update
            ## x_{n+1} = x_n - h{\nabla E_n(x_n) + \beta r_2[\nabla E_n(x_n+h/(1-\beta)**2 \nabla E_n(x_n)) -
            #  \nabla E_{n-1}(x_{n-1}+h/(1-\beta)**2 \nabla E_{n-1}(x_{n-1}))]} + \beta(x_n-x_{n-1})
            for p in group['params']:
                if p.grad is None:
                    continue
                first_term = self.state[p]["curr_grad"].clone()
                if iter > 0:
                    second_term = self.state[p]["pre_grad"].clone()
                self.state[p]["pre_grad"] = self.state[p]["curr_grad"].clone()
                d_p = self.state[p]["old_p_grad"].clone() 
                
                if iter > 0:                    
                    d_p = d_p.add(beta/(1-beta),first_term.add(-1,second_term))
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
                        buf = param_state['momentum_buffer'].clone()
                        buf.mul_(beta).add_(1 - dampening,d_p)
                    if nesterov:
                        d_p = d_p.add(beta,buf)
                    else:
                        d_p = buf
                
                p.data.add_(-1,d_p)
        return loss 

     
       
                    
                   
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        
        for group in self.param_groups:
            scale = group["t"]
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone() 
                self.state[p]["old_p_grad"] = p.grad.data.clone()
                p.add_(scale,p.grad.data)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            scale = group["t"]
            lr = group["lr"]
            beta = group['momentum']
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"   
                ext = p.grad.data.clone()    
                temp = self.state[p]["old_p_grad"].clone()
                temp1 = temp.add((lr/(1-beta)**2)*(1/scale),ext.add_(-1,temp)).clone()
                
                self.state[p]["curr_grad"] = temp1.clone()
                    
        
        if zero_grad: self.zero_grad()