from os import listdir
from os.path import isfile, join, normpath

import click
from PIL import Image
import numpy as np
from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.pretrained import pspnet_101_cityscapes

from ConvertCityscapesLabelColor import convert_image


def predict_pretrained_standard(model, inp, out_fname):
    model.predict_segmentation(inp, out_fname)


def predict_pretrained_split(model, inp, out_fname, split_count=4):
    image = Image.open(inp)
    out_image = Image.new("RGB", (image.width, image.height))

    for i in range(split_count):
        box = (int(i / split_count * image.width), 0, int((i + 1) / split_count * image.width), image.height)

        prediction = model.predict_segmentation(np.array(image.crop(box)), None)
        out_image.paste(convert_image(prediction).resize((int(image.width / split_count), image.height)), box[0:2])

    out_image.save(out_fname)


def predict_pretrained_croppped(model, inp, out_fname, box=(0, 110, 3340, 650)):
    image = Image.open(inp)
    out_image = Image.new("RGB", (image.width, image.height))

    prediction = model.predict_segmentation(np.array(image.crop(box)), None)

    out_image.paste(convert_image(prediction).resize((box[2] - box[0], box[3] - box[1])), box[0:2])
    out_image.save(out_fname)


@click.command()
@click.option('--images', '-i', default='./SYNTHIA-PANO/RGB/seqs02_fall/', help='The folder containing the images.', type=click.Path(exists=True))
@click.option('--predictions', '-p', default='./SYNTHIA-PANO/PREDICTIONS/seqs02_fall/', help='Where to store the predicted segmentation.', type=click.Path(exists=True))
@click.option('--pretrained/--from-checkpoint', default=True, help='Whether or not to use a pretrained model.', type=bool)
@click.option('--checkpoint_dir', help='Location of the model checkpoints. Only used with option --from-checkpoint.', type=click.Path(exists=True))
def predict(images, predictions, pretrained, checkpoint_dir):

    if pretrained:
        model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset
    else:
        model = model_from_checkpoint_path(normpath(checkpoint_dir))

    files = [f for f in listdir(images) if isfile(join(images, f))]

    for file in files:
        predict_pretrained_standard(model, inp=f'{images}{file}', out_fname=f'{predictions}{file}')


if __name__ == '__main__':
    predict()
