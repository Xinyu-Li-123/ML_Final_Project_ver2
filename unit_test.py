import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from models.definitions.vgg_nets import Vgg16Experimental
from utils import utils


# fig, axes = plt.subplots(1, 5)

def show_image(img, axes, index, title=None):
    axes[index].imshow(img)
    if title is not None:
        axes[index].set_title(title)

def augment_image(img, opcodes=("flip-vertical", 
                                "flip-horizontal", 
                                # "resize-0.4",
                                # "resize-0.2",
                                )):
    ops = {
        "flip": lambda img, direction: cv.flip(img, {"horizontal": 0, 
                                                     "vertical"  : 1}[direction]),  # flip
        "resize": lambda img, scale: cv.resize(img, (int(img.shape[1]*float(scale)), 
                                                     int(img.shape[0]*float(scale)))),
    }
    new_imgs = []
    for opcode in opcodes:
        op, code = opcode.split("-")
        print(op, code)
        new_imgs.append(ops[op](img, code))
    return new_imgs


def augment_img_tensor(img: torch.Tensor, opcodes=("flip-0", "flip-1", 
                                                   "crop-0.5",
                                                   "rotate-90", "rotate-180", "rotate-270")):
    np_img = img.numpy()

    def op(op_name: str, op_arg: int):
        if op_name == "flip":
            return cv.flip(np_img, op_arg)
        elif op_name == "crop":
            pass
        elif op_name == "rotate":
            return cv.rotate(np_img, 0)
        
    return


img_dir = "./data/content-images"
# img_name = "dark_forest.jpg"
img_name = "shop.jpg"
img_path = os.path.join(img_dir, img_name)

img = cv.imread(img_path)[:,:,::-1]

fig, axes = plt.subplots(1,4,figsize=(8,8))

show_image(img, axes, 0, "original image")
print(img.shape)

# for i, new_img in enumerate(augment_image(img)):
#     print(new_img.shape)
#     show_image(new_img, axes, i+1)
for code in range(3):
    show_image(cv.resize(cv.rotate(img, code), (img.shape[1], img.shape[0])), axes, code+1, title=str(code))

plt.show()

# model = Vgg16Experimental().to('cpu')
# img = utils.prepare_img(img_path, 400, device='cpu')
# out_1 = model(img).relu1_1
# out_2 = torch.tensor(cv.flip(out_1.detach().numpy(), 0), dtype=out_1.dtype)
# # print(np.linalg.norm(out_1.relu1_1-out_2.relu1_1))