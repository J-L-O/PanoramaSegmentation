from os import listdir
from os.path import isfile, join

from keras_segmentation.pretrained import pspnet_101_cityscapes

model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset

base_path = './SYNTHIA-PANO/RGB/seqs02_fall/'
out_path = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'

print(model.evaluate_segmentation(inp_images_dir="dataset1/images_prepped_test/", annotations_dir="dataset1/annotations_prepped_test/"))
