import torch
import torch.nn as nn


class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.1):
        super(MyLeakyReLU, self).__init__()
        if negative_slope < 0:
            raise ValueError("negative_slope should be >0, " "but got {}".format(negative_slope))
        self.negative_slope = negative_slope

    def forward(self, X_bottom):
        #######################################
        ## DO NOT CHANGE ANY CODE in forward ##
        #######################################
        # record the mask
        # TODO: Explain in hw why this is important ?
        self.mask = (X_bottom > 0)
        # print(self.mask)
        # slope is 1 for positive values
        mult_matrix = torch.ones_like(X_bottom)
        # negative_slope for negative values
        mult_matrix[~self.mask] = self.negative_slope
        X_top = X_bottom * mult_matrix
        return X_top

    def backward_manual(self, delta_X_top):
        # TODO: implement backward function
        # hint: re-use the recorded mask in forward() function
        delta_X_bottom = delta_X_top # modify this dummy line
        return delta_X_bottom


def main():
    ##################################
    ## DO NOT CHANGE ANY CODE BELOW ##
    ##      Explain TODO  places    ##
    ##################################
    '''
    Let y = leaky_relu(x) be prediction.
    Let the true value is 1.
    Then the loss L = (y-1.0)^2
    Delta_X = dL/dx = dL/dy * dy/dx = 2(y-1.0) * dy/dx
    Note that dy/dx is the backward_manual implemented by you
    We now compare your dy/dx with torch.autograd
    '''

    # test case as input
    x = torch.arange(-4, 5, dtype=torch.float32, requires_grad=True).view(3, 3)

    # =====================
    # == MyLeakyRelu ======
    # =====================
    my_leakyrelu = MyLeakyReLU(negative_slope=0.1)

    # forward
    print('Input ', x)
    y = my_leakyrelu.forward(x)
    print(' - my_relu forward:\n', y)

    # let's assume a toy model, with y = leaky_relu(x), loss = 0.5* y^2
    loss_y_0 = 0.5 * (y ** 2)
    # sum the loss to a scala
    loss_y = torch.sum(loss_y_0)

    # TODO: explain the result, what is dloss/dy
    y_diff = torch.autograd.grad(loss_y, y, retain_graph=True)[0]
    print('Loss y gradient is \n', y_diff)

    # Now we use two ways to compute dloss_y / dx, they should be the same
    if True:
        # TODO: explain the result, calculate the gradient with manual backward function you implemented
        dx = my_leakyrelu.backward_manual(y_diff)
        print('MyLeakyRelu manual backward:\n', dx)

        # TODO: explain the result, use torch autograd to get x's gradient
        dx2 = torch.autograd.grad(loss_y, x, retain_graph=True)[0]
        print('MyLeakyRelu auto backward:\n', dx2)

        # TODO: explain why dx=dx2, use chain rule to compute, then compare
        # hint: y = LeakyRelu(x), loss=0.5*y^2, by chain-rule, dy/dx = ?

    # =========================
    # == Torch LeakyRelu ======
    # =========================
    if True:
        print('\n========= Below is Pytorch Implementation ===========')
        # TODO: here we directly use Pytorch LeakyRelu. Explain, Should be y==y3? dx==dx3?
        torch_leakyrelu = torch.nn.LeakyReLU(negative_slope=0.1)
        # If not, you should check your implementation.
        y3 = torch_leakyrelu(x)
        print('Torch LeakyRelu forward:\n', y3)
        loss_y3 = torch.sum(0.5 * (y3 ** 2))
        dx3 = torch.autograd.grad(loss_y3, x, retain_graph=True)[0]
        print('Torch LeakyRelu manual backward:\n', dx3)

        # the assertions should be correct after your implementation
        assert torch.allclose(y, y3)
        assert torch.allclose(dx, dx3), 'the assertions should be correct after your implementation'


if __name__ == '__main__':
    main()
