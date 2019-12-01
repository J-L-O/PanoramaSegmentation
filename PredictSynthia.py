from os import listdir
from os.path import isfile, join

from PIL import Image
import numpy as np
from keras_segmentation.pretrained import pspnet_101_cityscapes

from ConvertCityscapesLabelColor import convert_image


def predict_pretrained_standard(inp, out_fname):
    model.predict_segmentation(inp, out_fname)


def predict_pretrained_split(inp, out_fname, split_count=4):
    image = Image.open(inp)
    out_image = Image.new("RGB", (image.width, image.height))

    for i in range(split_count):
        box = (int(i / split_count * image.width), 0, int((i + 1) / split_count * image.width), image.height)

        prediction = model.predict_segmentation(np.array(image.crop(box)), None)
        out_image.paste(convert_image(prediction).resize((int(image.width / split_count), image.height)), box[0:2])

    out_image.save(out_fname)


model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset

base_path = './SYNTHIA-PANO/RGB/seqs02_fall/'
out_path = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    predict_pretrained_split(inp=f'{base_path}{file}', out_fname=f'{out_path}{file.replace("_pano", "_label")}')
