import torch
import torch.nn.functional as F


# This code shows a stride-2 convolution can reduce image size by half

# constructs a matrix like in slide
D=torch.zeros(5,5).float()
D[1,1:4] += torch.arange(1,4)
D[2,1:4] += torch.arange(4,7)
D[3,1:4] += torch.arange(7,10)
print(D)
# we have to add two channels, since pytorch accepts (B, C, H, W)
# here we let B=1, C=1
D = D.unsqueeze(0).unsqueeze(0)

# Identity convolution kernal
H = torch.zeros(3,3).float()
H[1,1] = 1.0
print(H)
# same reason to make F.conv2d work
H =  H.unsqueeze(0).unsqueeze(0)


# should be same with D
out1 = F.conv2d(D, H, stride=[1,1], padding=[1,1])
print(out1)

# reduce D to half size, check padding
out2 = F.conv2d(D, H, stride=[2,2], padding=[1,1])
print(out2)