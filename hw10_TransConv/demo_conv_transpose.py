import torch
import torch.nn.functional as F

# This code shows a stride-2 transposed convolution can increase image size by twice
# Also, a TransConv can be learned reconstruct original data with SGD, as in a GAN model

# Let's build a data matrix
D = torch.arange(1,5).unsqueeze(0).repeat(4,1).float()
print(D)
D = D.unsqueeze(0).unsqueeze(0) # make it a batch

# Let's build an initial conv kernel
H = torch.ones(3,3).float()
print(H)
H =  H.unsqueeze(0).unsqueeze(0)

# Perform a standard conv
conv_out = F.conv2d(D, H, stride=1, padding=0)
print(conv_out)

# Can we simply conv out1 back to original data D ?
# Let's assume that H_inv can simply average the out1 back to D and check
H_inv = H / 9

# the padding option is a trick, read docs
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
# real_padding = 2
# param_padding = 1 * (3 (kernel_size) - 1) - real_padding = 0
param_padding = 0
inv_out_1 = F.conv_transpose2d(conv_out, H_inv, stride=1, padding=param_padding)
print(inv_out_1)

# similar, param_padding = 1 * (3 (kernel_size) - 1) - real_padding = 0
inv_out_2 = F.conv_transpose2d(conv_out, H_inv, stride=2, padding=0)
print(inv_out_2)

# TODO: Explain, are out2 and out3 close or deviating from original data D
# Is a simple average convolution can reconstruct D ?

# TODO: code the learnable TransConv below
# Can we learn a kernel to reconstruct D with SGD ? Let's do it.

# First we declare a tensor H_learnable
H_trans = torch.nn.Parameter(H_inv, requires_grad=True)

# We keep track of the best loss and best learned H_trans
loss_min = 1e10
H_best = None
num_epoch = 20
best_epoch = -1

# Then we set learning rate. You can modify lr freely
lr = 0.001

# Main loop of SGD
for epoch in range(num_epoch):
    # Trans_conv with H_trans
    out_cur = F.conv_transpose2d(conv_out, H_trans, stride=1, padding=0)
    # Use L2 distance to fit D
    loss = torch.mean((D-out_cur)**2)

    # TODO: backward the loss to get gradient of H_trans, add a line below

    # TODO: update H_trans data by subtracting the gradient * lr

    # DO NOT MODIFY
    print(out_cur, H_trans, loss.item())
    if loss_min > loss.item():
        loss_min = loss.item()
        H_best = H_trans.clone()
        best_epoch = epoch


print()
print('Best transpose conv is epoch %d \n' % (best_epoch,), H_best)
print('Best re-constructed data is \n', F.conv_transpose2d(conv_out, H_best, stride=1, padding=0))
print('Compare with original data\n', D)
