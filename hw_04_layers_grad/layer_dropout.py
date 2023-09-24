import torch
import torch.nn as nn

# Refer to Caffe Dropout implementation
# https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp
class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.scale = 1.0/(1-self.p)

    def forward(self, X):
        if self.training:
            # in training, randomly sample 1-prob. elements to be zero
            # check https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp#L39
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            # use the Binomial distribution to sample a binary mask
            # indicating which elements to drop (mask[i,j]=0) and retain (mask[i,j]=1)
            self.mask = binomial.sample(X.size())
            # dropout X
            X_mask = X * self.mask
            # then we have to scale the element values to be 1/(1-prob), to make them roughly sum to one
            # e.g., if p=0.5, you randomly dropout half elements in X, the left half of X should be made values as 2X
            X_scale = X_mask * self.scale
            return  X_scale
        
        # in inference (validation/testing), no need to scale
        return X

    def backward_manual(self, delta_X_top):
        if self.training:
            delta_X_bottom = delta_X_top * self.mask * self.scale
        else:
            delta_X_bottom = delta_X_top
        return delta_X_bottom


def main():
    '''
    Let y = relu(x) be prediction.
    Let the true value is 1.
    Then the loss L = (y-1.0)^2

    Delta_X = dL/dx = dL/dy * dy/dx = 2(y-1.0) * dy/dx

    '''

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    dropout = MyDropout(p=0.5)

    # test case, print out the input x
    x = torch.nn.Parameter(torch.arange(0,9).view(3,3).float(), requires_grad=True)

    # turn on training
    dropout.training = True

    # forward
    print('Input ', x)
    y = dropout.forward(x)
    print(' - dropout forward:\n', y)

    # let's assume a toy model, with y = dropout(x), loss = 0.5* y^2
    loss_y_0 = 0.5*(y**2)
    # sum the loss to a scala
    loss_y = torch.sum(loss_y_0)

    # TODO: explain the result, what is dloss/dy
    y_diff = torch.autograd.grad(loss_y, y, retain_graph=True)[0]
    print('Loss y gradient is \n', y_diff)

    # TODO: explain the result, use dropout manual backward function you implemented
    dx = dropout.backward_manual(y_diff)
    print('Dropout manual backward:\n', dx)

    # TODO: explain the result, use torch autograd to get x's gradient
    dx2 = torch.autograd.grad(loss_y, x, retain_graph=True)[0]
    print('Dropout auto backward:\n', dx2)

    # TODO: explain why dx=dx2, use chain rule to explain
    # hint: y = Dropout(x), loss=0.5*y^2, by chain-rule, dy/dx = ?




if __name__ == '__main__':
    main()