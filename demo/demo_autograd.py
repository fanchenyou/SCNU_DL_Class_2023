import torch
import torch.nn as nn


def main():

    # input x
    x = torch.tensor(1.0, requires_grad=False)
    # true label z=1
    z = torch.tensor(1.0, requires_grad=False)

    # param w
    w = torch.tensor(0.5, requires_grad=True)
    # param b
    b = torch.tensor(-1.0, requires_grad=True)

    optim = torch.optim.SGD([w,b], lr=0.1)


    # forward pass
    y = w*x+b
    print('Forward pass y=wx+b')
    # loss function
    loss_y = ((y-z) ** 2)
    print('We got y=',y.item(),'  loss=', loss_y.item())

    # directly use autograd
    y_diff = torch.autograd.grad(loss_y, w, retain_graph=True)[0]
    print('Directly use autograd')
    print('Gradient of Loss w.r.t w is \n', y_diff)

    # use an optimizer to check the gradient
    optim.zero_grad()
    loss_y.backward()
    print('Use Pytorch Optimizer')
    print('Gradient of Loss w.r.t w is \n', w.grad)
    optim.step()
    print('Apply SGD with learning rate', optim.param_groups[0]['lr'])
    print(w.data, ' equals to w-lr*grad = 0.5-0.1*(-3)=0.8')




if __name__ == '__main__':
    main()
