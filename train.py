import os

import click
from keras_segmentation.models.pspnet import pspnet_101


@click.command()
@click.option('--images', '-i', default='./SYNTHIA-PANO/RGB/seqs04_fall/', help='The folder containing the images.', type=click.Path(exists=True))
@click.option('--labels', '-l', default='./SYNTHIA-PANO/LABELS/seqs04_fall/', help='The folder containing the ground truth segmentation.', type=click.Path(exists=True))
@click.option('--width', '-w', default=835, help='The image width.', type=int)
@click.option('--height', '-h', default=190, help='The image height.', type=int)
def train_pspnet_101(images, labels, width, height):
    model = pspnet_101(n_classes=16, input_width=width, input_height=height)

    print('Start training')

    checkpoint_directory = f'./checkpoints/pspnet101_{width}x{height}'
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    model.train(
        train_images=images,
        train_annotations=labels,
        checkpoints_path=f'./checkpoints/pspnet101_{width}x{height}/pspnet101_{width}x{height}', epochs=5, batch_size=1
    )

    return model


if __name__ == '__main__':
    train_pspnet_101()

