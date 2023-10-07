import torch
import cv2
import torchvision.transforms as transforms

# install opencv properly
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python

# Read the image
image = cv2.imread('bear.png')

# Convert BGR image to RGB image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define a transform to convert
# the image to torch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Convert the image to Torch tensor
img_tensor = transform(image)

# print the converted image tensor
print(img_tensor.size())
print(img_tensor)