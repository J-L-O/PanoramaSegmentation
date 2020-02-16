import os

import click
from keras_segmentation.models.pspnet import pspnet_101, resnet50_pspnet
from keras_segmentation.models.segnet import resnet50_segnet
from keras_segmentation.models.unet import resnet50_unet


@click.command()
@click.option('--images', '-i', default='./SYNTHIA-PANO/RGB/seqs04_fall/', help='The folder containing the images.', type=click.Path(exists=True))
@click.option('--labels', '-l', default='./SYNTHIA-PANO/LABELS/seqs04_fall/', help='The folder containing the ground truth segmentation.', type=click.Path(exists=True))
@click.option('--model', '-m', default='pspnet_101', help='The model to train.', type=str)
@click.option('--width', '-w', default=835, help='The image width.', type=int)
@click.option('--height', '-h', default=190, help='The image height.', type=int)
def train(images, labels, model, width, height):

    if model == 'pspnet_101':
        segmentation_model = pspnet_101(n_classes=16, input_width=width, input_height=height)
    elif model == 'resnet50_unet':
        segmentation_model = resnet50_unet(n_classes=16, input_width=width, input_height=height)
    elif model == 'resnet50_segnet':
        segmentation_model = resnet50_segnet(n_classes=16, input_width=width, input_height=height)
    elif model == 'resnet50_pspnet':
        segmentation_model = resnet50_pspnet(n_classes=16, input_width=width, input_height=height)
    else:
        return None

    print('Start training')

    checkpoint_directory = f'./checkpoints/{model}_{width}x{height}'
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    segmentation_model.train(
        train_images=images,
        train_annotations=labels,
        checkpoints_path=f'{checkpoint_directory}/{model}_{width}x{height}', epochs=10, batch_size=1
    )

    return segmentation_model


if __name__ == '__main__':
    train()

