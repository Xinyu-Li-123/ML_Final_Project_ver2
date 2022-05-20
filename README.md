# Machine Learning Final Project

### About

This is a fork of [gordicaleksa's implementation of Neural Style Transfer](https://github.com/gordicaleksa/pytorch-neural-style-transfer). I added two features to gordicaleksa's original code: **colorspace conversion** and **content augmentation**.

To test out the model, you can use the following command in the powershell
```cmd
!python neural_style_transfer.py --content_img_name shop.jpg --style_img_name ligne.jpg ^
--content_weight 1e5 --style_weight 3e4 --tv_weight 1e0 ^
--loss_display_freq 50 --saving_freq 100 --max_iter 300 --height 400 ^
--preserve_color 1 --pretrained 1 --aug_type 0 ^
--pooling_method avg --init_method content --reshape_method resize ^
--style_feature_maps_indices 0_2_4_6 --content_feature_maps_indices 6
```

