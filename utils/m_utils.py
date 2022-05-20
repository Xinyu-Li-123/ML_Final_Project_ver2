#coding:utf-8

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt

try:
    from models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental
except ModuleNotFoundError:
    pass

# magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
num_of_iterations = {
    "lbfgs": 300,
    "adam": 3000,
}


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

#
# Image manipulation util functions
#

DEFAULT_HEIGHT = 400
# using the FCC NTSC Standard
RGB2YIQ = np.array(
    [[0.3  ,  0.59  ,  0.11  ],
     [0.599, -0.2773, -0.3217],
     [0.213, -0.5251,  0.3121]])
YIQ2RGB = np.array(
    [[1,  0.9469,  0.6236],
     [1, -0.2748, -0.6357],
     [1, -1.1   ,  1.7   ]])

def load_image(img_path, target_shape=None):
    """
    Load image from img_path, reshape to target_shape
    img.shape = (H, W, C)
    """
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    
    # enable cv2 to read chinese
    def cv_imread(file_path):
        cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img
    img = cv_imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img: np.ndarray, target_shape, device, type="rgb", should_aug=False):
    """
    img: img.shape = (H, W, C)
    
    return torch.Tensor() with shape (N, C, H, W)
    """
    # img = load_image(img_path, target_shape=target_shape)
    if type == "yiq":
        img = rgb2yiq(img)
        # img = np.moveaxis(img, 2, 0)    
        luminance_img = img[:,:,0]
        color_img = img[:,:,1:]
        luminance_img = y2rgb(luminance_img)        # y channel -> rgb colorspace        
        luminance_img = luminance_img.astype(np.float32)    
        
    # normalize using ImageNet's mean
    # [0, 255] range worked much better for me than [0, 1] range (even though PyTorch models were trained on latter)    
    # Note that, should aug or not, we will always unsqueeze a dimension at axis=0 by calling torch.stack
    if should_aug:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),     # normalize and move axis
            transforms.Lambda(lambda x: torch.stack([               # augment image
                x,
                TF.hflip(x),
                TF.vflip(x),
                TF.rotate(x, 90.0),
                TF.rotate(x, 180.0),  
                TF.rotate(x, 270.0),
                # TF.rotate(TF.hflip(x), 90.0),
                # TF.rotate(TF.vflip(x), 90.0),
            ]))
        ])
    else: 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL),     # normalize and move axis
            transforms.Lambda(lambda x: torch.stack([
                x,
            ]))   
        ])

    if type == "rgb":
        img = transform(img).to(device)    # ndarray -> tensor
        return img
    elif type == "yiq":
        luminance_img = transform(luminance_img).to(device)    # ndarray -> tensor
        # in this case, the return type are as follow
        #       luminance_img: torch.Tensor, shape=[N, C, H, W]
        #       color_img: numpy.ndarray, shape=[H, W, C]
        # luminamce img is used for the optimization
        # color_img is used to recreate colored image from the output image of the optimization process.
        return luminance_img, color_img


def rgb2yiq(img: np.ndarray):
    """ 
    img.shape = (H, W, C)
    c: channel (R, G, B) 
    """
    new_img = img.copy().reshape(-1, 3) @ RGB2YIQ
    new_img = new_img.reshape(img.shape)
    return new_img
    # return (new_img-np.min(new_img))/(np.max(new_img) - np.min(new_img))    # rescale to [0, 1]


def yiq2rgb(img: np.ndarray):
    """
    img: np.ndarray
        img.shape = (h, w, c)
        where c: channel_num (Y, I, Q) 
    handle_overflow: str
        if handle_overflow == truncate:
            truncate to range [0, 1]
        elif handle_overflow == rescale:
            rescale to range [0, 1]
    """
    new_img = img.copy().reshape(-1, 3) @ YIQ2RGB

    # # truncate the image to range [0, 1]
    # mask_1 = new_img > 1
    # new_img[mask_1] = 1     # 1.001 => 1.0
    # mask_0 = new_img < 0
    # new_img[mask_0] = 0     # -0.005 => 0.0

    # new_img = new_img.reshape(img.shape)

    new_img = np.clip(new_img, 0, 1)
    new_img = new_img.reshape(img.shape)
    return new_img

def y_iq_2rgb(y_img:np.ndarray, iq_img: np.ndarray):
    """
    combine y channel and iq channel, convert to rgb colorspace
    y_img: nd.array
        y channel, luminance image
        y_img.shape = (h, w)
    iq_img: nd.array
        iq channel, color image
        iq_img.shape = (h, w, 2)
    """
    img_shape = np.array([0, 0, 3])
    img_shape[0:2] = y_img.shape        # (h, w, c)
    new_img = np.zeros(img_shape)
    new_img[:,:,0] = y_img
    new_img[:,:,1:] = iq_img            # concatenate y, iq channels to yiq 
    return yiq2rgb(new_img)


def y2rgb(img: np.ndarray):
    """
    Convert y channel in yiq colorspace to rgb colorspace via following steps:
    -   separate y channel from yiq, 
    -   truncate y channel to [0, 1]
    -   convert grayscale to rgb
    The is meant to feed y channel into the cnn (pretrained model only accept rgb channel image)

    img: np.ndarray
        y channel of yiq
        img.shape = (h, w)
    """
    new_img = img

    # truncate to [0, 1]
    mask_1 = new_img > 1
    new_img[mask_1] = 1     # 1.001 => 1.0
    mask_0 = new_img < 0
    new_img[mask_0] = 0     # -0.005 => 0.0

    new_img = np.array([new_img, new_img, new_img])      

    return np.rollaxis(new_img, 0, 3)


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

    def resize(img: np.ndarray, target_shape):
        return cv.resize(img, target_shape)

    # aug_imgs = {
    #     "original": img, 
    #     "r0": rotate(img, 0),
    #     "r1": rotate(img, 1),
    #     "r2": rotate(img, 2),
    #     "f0": flip(img, 0),
    #     "f1": flip(img, 1),
    #     "r0f0": rotate(flip(img, 0), 0),
    #     "r0f1": rotate(flip(img, 1), 0),
    # }
    aug_imgs = [
        img,    # original 
        rotate(img, 0),   # r0
        rotate(img, 1),   # r1
        rotate(img, 2),   # r2
        flip(img, 0),     # f0
        flip(img, 1),     # f1
        # rotate(flip(img, 0), 0),    # r0f0
        # rotate(flip(img, 1), 0),    # r1f1
    ]
    # resize to the same shape as the original image
    for i in range(len(aug_imgs)):
        aug_imgs[i] = resize(aug_imgs[i], target_shape=(img.shape[1], img.shape[0]))
    for img in aug_imgs:
        print(img.shape, img.dtype)

    aug_imgs = np.asarray(aug_imgs)
    
    return aug_imgs

def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])  # [:, :, ::-1] converts rgb into bgr (opencv contraint...)


def generate_out_img_name(config):
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    # called from the reconstruction script
    if 'reconstruct_script' in config:
        suffix = f'_o_{config["optimizer"]}_h_{str(config["height"])}_m_{config["model"]}{config["img_format"][1]}'
    else:
        suffix = f'_o_{config["optimizer"]}_i_{config["init_method"]}_h_{str(config["height"])}_m_{config["model"]}_cw_{config["content_weight"]}_sw_{config["style_weight"]}_tv_{config["tv_weight"]}{config["img_format"][1]}'
    return prefix + suffix


def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations, should_display=False):
    saving_freq = config['saving_freq']
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr


    # for saving_freq == -1 save only the final result (otherwise save with frequency saving_freq and save the last pic)
    # if img_id == num_of_iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
    img_format = config['img_format']
    out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
    
    dump_img = np.copy(out_img)
    if config["preserve_color"]:
        # print(f"line 190: out_img : mean: {out_img.mean()}, var: {out_img.var()}, [{out_img.min()}, {out_img.max()}]")
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3)) 
        dump_img /= 255        # scale to [0, 1] to combine luminance and color
        dump_luminance_img = dump_img[:,:,:].mean(axis=2)
        dump_color_img = config['content_color_img']
        # print(f"luminance: shape={dump_luminance_img.shape}, min={dump_luminance_img.min() :.2f}, max={dump_luminance_img.max() :.2f}")
        # print(f"color: shape={dump_color_img.shape}, min={dump_color_img.min() :.2f}, max={dump_color_img.max() :.2f}")
        dump_img = y_iq_2rgb(dump_luminance_img, dump_color_img) * 255
    else:
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    dump_img = np.clip(dump_img, 0, 255).astype('uint8')

    cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
    print(f"{os.path.join(dump_path, out_img_name)} is saved")

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


#
# End of image manipulation util functions
#


# initially it takes some time for PyTorch to download the models into local cache
def prepare_model(model, device, pretrained=True, pooling_method='max', content_feature_maps_indices=None, style_feature_maps_indices=None):
    # we are not tuning model weights -> we are only tuning optimizing_img's pixels! (that's why requires_grad=False)
    experimental = True
    if model == 'vgg16':
        if experimental:
            # much more flexible for experimenting with different style representations
            print("using vgg16 experimental")
            model = Vgg16Experimental(requires_grad=False, 
                                      show_progress=True, 
                                      pretrained=pretrained, 
                                      pooling_method=pooling_method,
                                      content_feature_maps_indices=content_feature_maps_indices,
                                      style_feature_maps_indices=style_feature_maps_indices)
            print(f"pretrained: {pretrained}")
        else:
            model = Vgg16(requires_grad=False, show_progress=True, pretrained=pretrained)
    elif model == 'vgg19':
        model = Vgg19(requires_grad=False, show_progress=True, pretrained=pretrained)
    else:
        raise ValueError(f'{model} not supported.')

    # content_feature_maps_index = model.content_feature_maps_index
    content_feature_maps_indices = model.content_feature_maps_indices
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names

    # content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    content_fm_indices_names = (content_feature_maps_indices, layer_names) 
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fm_indices_names, style_fms_indices_names

def content_repr(x):
    pass

def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)     # covariance between channels
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

if __name__ == "__main__":
    img_dir = "./data/content-images"
    img_name = "shop.jpg"
    img_path = os.path.join(img_dir, img_name)
    # img = prepare_img(img_path, "400", device="cpu")
    img = load_image(img_path, target_shape=(400, 500))
    print(type(img), img.shape, img.dtype)
    should_aug_content = True
    
    luminance_img, color_img = prepare_img(img, (400, 500), "cpu", type="yiq", should_aug=should_aug_content)
    print(luminance_img.shape, type(luminance_img))
    print(color_img.shape, type(color_img))
