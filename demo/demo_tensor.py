import torch

torch.manual_seed(123)

#######################
# make a tensor on
#  cpu torch.float32
#######################
print('--------------------------- initialize')

# tensor([[0.2961, 0.5166, 0.2517],
#         [0.6886, 0.0740, 0.8665],
#         [0.1366, 0.1025, 0.1841]])
# cpu
# torch.float32
x = torch.rand(3, 3).float()
print(x, x.device, x.dtype)

# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]) cpu torch.int64
y = torch.arange(9)
print(y, y.device, y.dtype)
# torch.Size([9])
print(y.size())

# change shape to torch.Size([3, 3])
y = y.view(3, 3)
print(y.size())

#########
# mult
#########
print('--------------------------- multiply')
print(y * y)
print(torch.mm(y, y))

#########
# slice
#########
print('--------------------------- slice')

# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])
print(y)

# tensor([3, 4, 5])
print(y[1, :])

# tensor([2, 5, 8])
print(y[:, 2])

#############
# bool index
#############
print('--------------------------- index')

index = y >= 4

# tensor([[False, False, False],
#         [False,  True,  True],
#         [ True,  True,  True]])
print(index)

# tensor([4, 5, 6, 7, 8])
print(y[index])

indices = torch.tensor([0, 2])

# tensor([[0, 1, 2],
#         [6, 7, 8]])
print(torch.index_select(y, 0, indices))

# tensor([[0, 2],
#         [3, 5],
#         [6, 8]])
print(torch.index_select(y, 1, indices))

#############
# reshape
#############
print('--------------------------- reshape')
print(y.size())
z0 = y.view(9, 1)
print(z0.size())

# torch.Size([1, 3, 3])
z1 = y.unsqueeze(0)
print(z1.size())

# torch.Size([3, 3, 1])
z2 = y.unsqueeze(-1)
print(z2.size())

#############
# transpose
#############
# tensor([[0, 3, 6],
#         [1, 4, 7],
#         [2, 5, 8]])
print('--------------------------- transpose')
z = y.t()
print(z)

#######################################
# concatenate multiple matrices
#######################################
print('--------------------------- concatenate')

# tensor([[[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]],
#
#         [[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]]])
# torch.Size([2, 3, 3])

y2 = [y.unsqueeze(0), y.unsqueeze(0)]
y3 = torch.cat(y2, dim=0)
print(y3, '\n', y3.shape)

# tensor([[[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]],
#
#         [[0, 1, 2],
#          [3, 4, 5],
#          [6, 7, 8]]])
# torch.Size([2, 3, 3])

z2 = [y, y]
z3 = torch.stack(z2, dim=0)
print(z3, '\n', z3.size())

# tensor([[[0, 1, 2],
#          [0, 1, 2]],
#
#         [[3, 4, 5],
#          [3, 4, 5]],
#
#         [[6, 7, 8],
#          [6, 7, 8]]])
# torch.Size([3, 2, 3])

z4 = torch.stack(z2, dim=1)
print(z4, '\n', z4.size())

print('--------------------------- split')


# (tensor([[0, 1, 2],
#         [3, 4, 5]]),
#  tensor([[6, 7, 8]]))
y3s = torch.split(y, 2, dim=0)
print(y3s)

# (tensor([[0, 1, 2]]), tensor([[3, 4, 5]]), tensor([[6, 7, 8]]))
y3s = torch.chunk(y, 3, dim=0)
print(y3s)
