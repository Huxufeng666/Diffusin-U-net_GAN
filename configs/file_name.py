import os
import datetime

def get_save_paths(args):
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    mode_to_tag = {
        'diffusion_only': 'Diffusion',
        'unet_only': 'UNet',
        'finetune_unet': 'UNet_Finetune',
        'full_pipeline': 'FullPipeline'
    }
    model_tag = mode_to_tag.get(args.train_mode, 'Unknown')

    weights_dir = os.path.join(args.weights_dir, f"{formatted_time}_{model_tag}")
    os.makedirs(weights_dir, exist_ok=True)

    csv_file = os.path.join(weights_dir, f"training_{model_tag}_log.csv")
    return weights_dir, csv_file
