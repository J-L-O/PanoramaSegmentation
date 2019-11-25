from os import listdir
from os.path import isfile, join
from PIL import Image
from cvt2color import cvt2color
import numpy as np

base_path = './SYNTHIA-PANO/LABELS/seqs02_fall/'
out_path = './SYNTHIA-PANO/LABELS_Converted/seqs02_fall/'

color_dict_SYNTHIA_label_to_color = {
    0: (0, 0, 0),  # void
    1: (128, 128, 128),  # sky
    2: (128, 0, 0),  # building
    3: (128, 64, 128),  # road
    4: (0, 0, 192),  # sidewalk
    5: (64, 64, 128),  # fence
    6: (128, 128, 0),  # vegetation
    7: (192, 192, 128),  # pole
    8: (64, 0, 128),  # car
    9: (192, 128, 128),  # traffic sign
    10: (64, 64, 0),  # pedestrian
    11: (0, 128, 192),  # bicycle
    12: (0, 172, 0),  # landmarking
    13: (0, 0, 0),  # void
    14: (0, 0, 0),  # void
    15: (0, 128, 128)  # traffic light
}

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    img = Image.open(f'{base_path}{file}').convert('L')
    out = cvt2color(np.array(img), color_dict=color_dict_SYNTHIA_label_to_color)
    converted = Image.fromarray(out)
    converted.save(f'{out_path}{file}')
