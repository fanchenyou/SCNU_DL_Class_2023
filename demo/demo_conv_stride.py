import torch
import torch.nn.functional as F


# This code shows a stride-2 convolution can reduce image size by half


D=torch.zeros(5,5).float()
D[1,1:4] += torch.arange(1,4)
D[2,1:4] += torch.arange(4,7)
D[3,1:4] += torch.arange(7,10)
print(D)

D = D.unsqueeze(0).unsqueeze(0)

H = torch.zeros(3,3).float()
H[1,1] = 1.0
print(H)

H =  H.unsqueeze(0).unsqueeze(0)



out1 = F.conv2d(D, H, stride=[1,1], padding=[1,1])
print(out1)

out2 = F.conv2d(D, H, stride=[2,2], padding=[1,1])
print(out2)