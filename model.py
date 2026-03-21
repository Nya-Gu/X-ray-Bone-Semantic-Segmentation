import segmentation_models_pytorch as smp

def get_model(model_config, target_class):
    model_name      = model_config['name']
    model_encoder   = model_config['encoder']
    pretrained      = model_config['pretrained']
    in_channels     = model_config['in_channels']
    # classes         = model_config['classes']

    class_num_dict = {"all": 29, "finger": 19, "wrist": 8, "arm": 2, "mask": 4}
    classes         = class_num_dict[target_class]

    if model_name == 'Unet':
        model = smp.Unet(
        encoder_name = model_encoder,
        encoder_weights = pretrained,
        in_channels = in_channels,
        classes = classes,
        )
        print(f"학습 모델: Unet + {model_encoder}")
    elif model_name == "Segformer":
        model = smp.Segformer(
        encoder_name = model_encoder,
        encoder_weights = pretrained,
        in_channels = in_channels,
        classes = classes,
        )
        print(f"학습 모델: Segformer + {model_encoder}")
    elif model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
        encoder_name = model_encoder,
        encoder_weights = pretrained,
        in_channels = in_channels,
        classes = classes,
        )
        print(f"학습 모델: UnetPlusPlus + {model_encoder}")
    else:
        assert "모델은 Unet, Segforemr, UnetPlusPlus 중 하나를 사용해주시길 바랍니다."
    
    return model