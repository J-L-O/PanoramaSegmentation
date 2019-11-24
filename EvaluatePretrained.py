from os import listdir
from os.path import isfile, join

from keras_segmentation.pretrained import pspnet_101_cityscapes

model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset

base_path = './SYNTHIA-PANO/RGB/seqs02_fall/'
out_path = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    out = model.predict_segmentation(inp=f'{base_path}{file}', out_fname=f'{out_path}{file}')
