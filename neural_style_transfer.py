from multiprocessing import pool
import utils.utils as utils
from utils.utils import num_of_iterations
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse
from models.definitions.vgg_nets import Vgg16Experimental

# print("Randomly Initializing Content VGG...")
# content_vgg = Vgg16Experimental(pretrained=False)
# print("Content vgg initialized successfully")

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    # current_content_representation = content_vgg(optimizing_img)[content_feature_maps_index].squeeze(axis=0)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)    

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    # content_loss = style_loss - style_loss

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config, content_img_name: str, style_img_name: str):
    
    content_img_path = os.path.join(config['content_images_dir'], content_img_name)
    style_img_path = os.path.join(config['style_images_dir'], style_img_name)

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # content_vgg.to(device)
    # print(f"content vgg is using device: {device}")

    if not config["preserve_color"]:
        content_img = utils.prepare_img(content_img_path, config['height'], device)
        style_img = utils.prepare_img(style_img_path, config['height'], device)
    else:
        content_img, content_color_img = utils.prepare_img(content_img_path, config['height'], device, type="yiq")
        style_img, style_color_img = utils.prepare_img(style_img_path, config['height'], device, type="yiq")
        config["content_color_img"] = content_color_img     # if preserve color, save color_img to config
        config["style_color_img"] = style_color_img     


    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)


    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], 
                                                                                                        device, 
                                                                                                        pretrained=config['pretrained'],
                                                                                                        pooling_method=config['pooling_method'])
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    

    style_img_set_of_feature_maps = neural_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]

    # style_img_set_of_feature_maps = torch.tensor(neural_net(style_img))
    # aug_style_images = utils.augment_image(style_img)       # a list of augmented style images
    # for style_img in aug_style_images:
    #     style_img_set_of_feature_maps += torch.tensor(neural_net(style_img))
    # style_img_set_of_feature_maps /= len(aug_style_images)  # take average over style images
        
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    #
    # Start of optimization procedure
    #
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                if cnt % 50 == 0:
                    print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.png')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='shop.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='candy.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=utils.DEFAULT_HEIGHT)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e1)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='lbfgs')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg16')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--pooling_method", type=str, choices=['max', 'avg'], help="pooling method, max pooling or average pooling", default='max')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=2)
    parser.add_argument("--max_iter", type=int, help="maximal number of iterations", default=300)
    parser.add_argument("--preserve_color", type=int, help="whether or not to preserve the color of the content image", default=0)
    parser.add_argument("--pretrained", type=int, help="whether or not to use a pretrained model", default=1)
    parser.add_argument("--content_augmented", type=int, help="whether or not to augment the dataset", default=0)
    parser.add_argument("--style_augmented", type=int, help="whether or not to augment the dataset", default=0)
        


    args = parser.parse_args()

    # some values of weights that worked for figures.jpg, vg_starry_night.jpg (starting point for finding good images)
    # once you understand what each one does it gets really easy -> also see README.md

    # lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
    # lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
    # lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

    # adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
    # adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
    # adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format
    optimization_config['preserve_color'] = bool(optimization_config['preserve_color'])
    optimization_config['pretrained'] = bool(optimization_config['pretrained'])
    num_of_iterations[optimization_config['optimizer']] = optimization_config['max_iter']


    print(f"images:\n"
          f"    content image: {optimization_config['content_img_name']}\n"
          f"    style image  : {optimization_config['style_img_name']}\n")
    print(f"weights:\n"
          f"    style_weight  : {optimization_config['style_weight']:.1}\n"
          f"    content_weight: {optimization_config['content_weight']:.1}\n"
          f"    tv_weight     : {optimization_config['tv_weight']:.1}")
    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    print(f"init method: {optimization_config['init_method']}\n"
          f"preserve_color: {optimization_config['preserve_color']}\n"
          f"optimizer: {optimization_config['optimizer']}\n"
          f"num of iterations: {num_of_iterations[optimization_config['optimizer']]}")
    # print(f"pretrained    : {optimization_config['pretrained']} ")


    # expose img_name to iterate through augmented images

    content_img_name = optimization_config["content_img_name"]
    style_img_name = optimization_config["style_img_name"]

    if optimization_config["content_augmented"] + optimization_config["style_augmented"] == 2:
        raise ValueError("You can only augment either content or style")
    
    print(optimization_config["content_augmented"], optimization_config["style_augmented"])
    if optimization_config["content_augmented"]:
        # iterate over augmented content images
        print("augmented type: content")
        aug_content_img_names = utils.augment_img(optimization_config, aug_type="content")
        for aug_content_img_name in aug_content_img_names:
            neural_style_transfer(optimization_config, aug_content_img_name, style_img_name)
            print(f"{aug_content_img_name} completed")

    elif optimization_config["style_augmented"]:
        print("augmented type: content")
        pass
    else:
        print("augmented type: None")
        neural_style_transfer(optimization_config, content_img_name, style_img_name)
        pass
    


    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)

    