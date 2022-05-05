import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


def augment_img(img: np.ndarray):
    def flip(img: np.ndarray, code: int):
        if code in (0,1):
            return cv.flip(img, code)
        else:
            raise ValueError(f"code must be 0, 1, or 2, instead of {code}")

    def rotate(img: np.ndarray, code: int):
        if code in (0,1,2):
            return cv.rotate(img, code)
        else:
            raise ValueError(f"code must be 0, 1, or 2, instead of {code}")

    # imgs = [
    #     img, 
    #     rotate(img, 0),
    #     rotate(img, 1),
    #     rotate(img, 2),
    #     flip(img, 0),
    #     flip(img, 1),
    #     rotate(flip(img, 0), 0),
    #     rotate(flip(img, 1), 0),
    # ]


    imgs = {
        "original": img, 
        # "r0": rotate(img, 0),
        # "r1": rotate(img, 1),
        # "r2": rotate(img, 2),
        # "f0": flip(img, 0),
        # "f1": flip(img, 1),
        # "r0f0": rotate(flip(img, 0), 0),
        # "r0f1": rotate(flip(img, 1), 0),
    }

    for op_name in imgs:
        print(op_name)
        cv.imwrite(os.path.join(img_dir, op_name+"_"+img_name), imgs[op_name])
    
    return imgs

img_dir = "./data/content-images/"
img_name = "green_bridge.jpg"

img = cv.imread(os.path.join(img_dir, img_name))

imgs = augment_img(img)

for img in imgs:
    print(img.shape)


# fig, axes = plt.subplots(2,4,figsize=(8,8))

# axes[0][0].imshow(img)

# axes[0][1].imshow(rotate(img, 0))
# axes[0][2].imshow(rotate(img, 1))
# axes[0][3].imshow(rotate(img, 2))

# axes[1][0].imshow(flip(img, 0))
# axes[1][1].imshow(flip(img, 1))

# axes[1][2].imshow(rotate(flip(img, 0), 0))
# axes[1][3].imshow(rotate(flip(img, 1), 0))

# plt.show()