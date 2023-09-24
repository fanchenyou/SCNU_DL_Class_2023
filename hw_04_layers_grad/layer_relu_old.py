import torch
import torch.nn as nn

class MyLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(MyLeakyReLU, self).__init__()
        if negative_slope < 0:
            raise ValueError("negative_slope should be >0, " "but got {}".format(negative_slope))
        self.negative_slope = negative_slope

    def forward(self, X_bottom):
        # record the mask, why this is important ?
        mask = (X_bottom > 0)
        # you can check the mask by printing it out
        # print(mask)
        self.mask = mask
        # slope is 1 for positive values
        mult_matrix = torch.ones_like(X_bottom)
        # negative_slope for negative values
        mult_matrix[~mask] = self.negative_slope
        X_top = X_bottom * mult_matrix                    
        return  X_top
        
    
    def backward(self, delta_X_top):
        mult_matrix = torch.ones_like(delta_X_top)
        mult_matrix[~self.mask] = self.negative_slope
        delta_X_bottom = delta_X_top * mult_matrix
        return delta_X_bottom

    

def main():
    '''
    Let y = relu(x) be prediction.
    Let the true value is 1.
    Then the loss L = (y-1.0)^2

    Delta_X = dL/dx = dL/dy * dy/dx = 2(y-1.0) * dy/dx

    '''
    relu = MyLeakyReLU(negative_slope=0.05)

    # test case, scalar
    x0 = torch.tensor([1])
    x1 = torch.tensor([-1.5])
    # test case, 1-D tensor
    x2 = torch.tensor([1, -1])
    x3 = torch.tensor([-1.5, 0])
    # test case, 2-D tensor
    x4 = torch.tensor([[1.5, -1],[100,1]])
    x5 = torch.tensor([[-1.5, 1],[-100,-1]])

    for i, x in enumerate([x0,x1,x2,x3,x4,x5]):
        print('Input x%d\n' % (i,), x)
        y = relu.forward(x)
        print(' - relu forward:\n', y)
        dy = 2.0*(y-1.0)
        dx = relu.backward(dy)
        print(' - relu backward:\n', dx)
        print()

    
    
    # Questions:
    # 1. make sure your output is identical to the correct answer below
    # 2. Why x0 has gradient 0 (no error) ? 
    # 3. Following Q2, when an input x has gradient 0.
    # 4. Following Q3, when an input x has large gradient (large error)? When small gradient (small error) ?
    # 5. Following Q4, check x4 and x5 output. 
    #    Whether a large positive (e.g., 100) or a large negative (e.g., -100) result in a more error ?


    '''
    Input x0
    tensor([1])
    - relu forward:
    tensor([1])
    - relu backward:
    tensor([0.])

    Input x1
    tensor([-1.5000])
    - relu forward:
    tensor([-0.0750])
    - relu backward:
    tensor([-0.1075])

    Input x2
    tensor([ 1, -1])
    - relu forward:
    tensor([1, 0])
    - relu backward:
    tensor([ 0.0000, -0.1000])

    Input x3
    tensor([-1.5000,  0.0000])
    - relu forward:
    tensor([-0.0750,  0.0000])
    - relu backward:
    tensor([-0.1075, -0.1000])

    Input x4
    tensor([[  1.5000,  -1.0000],
            [100.0000,   1.0000]])
    - relu forward:
    tensor([[ 1.5000e+00, -5.0000e-02],
            [ 1.0000e+02,  1.0000e+00]])
    - relu backward:
    tensor([[ 1.0000e+00, -1.0500e-01],
            [ 1.9800e+02,  0.0000e+00]])

    Input x5
    tensor([[  -1.5000,    1.0000],
            [-100.0000,   -1.0000]])
    - relu forward:
    tensor([[-0.0750,  1.0000],
            [-5.0000, -0.0500]])
    - relu backward:
    tensor([[-0.1075,  0.0000],
            [-0.6000, -0.1050]])
    '''




if __name__ == '__main__':
    main()