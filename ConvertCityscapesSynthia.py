from os import listdir
from os.path import isfile, join
from PIL import Image
from cvt2color import cvt2color
import numpy as np

base_path = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'
out_path = './SYNTHIA-PANO/PREDICTIONS_Converted/seqs02_fall/'

color_dict_CITYSCAPES_to_SYNTHIA_color = {
    (20, 215, 197): (128, 64, 128),  # road -> road
    (207, 248, 132): (0, 0, 192),  # sidewalk -> sidewalk
    (183, 244, 155): (128, 0, 0),  # building -> building
    (144, 71, 111): (128, 0, 0),  # wall -> building
    (128, 48, 71): (64, 64, 128),  # fence -> fence
    (50, 158, 75): (192, 192, 128),  # pole -> pole
    (241, 169, 37): (0, 128, 128),  # traffic light -> traffic light
    (222, 181, 51): (192, 128, 128),  # traffic sign -> traffic sign
    (244, 104, 161): (128, 128, 0),  # vegetation -> vegetation
    (31, 133, 226): (0, 0, 0),  # terrain -> void
    (204, 47, 7): (128, 128, 128),  # sky -> sky
    (170, 252, 0): (64, 64, 0),  # person -> pedestrian
    (32, 166, 124): (0, 128, 192),  # rider -> bicycle
    (122, 113, 97): (64, 0, 128),  # car -> car
    (46, 229, 72): (64, 0, 128),  # truck -> car
    (250, 163, 41): (64, 0, 128),  # bus -> car
    (149, 154, 55): (64, 0, 128),  # train -> car
    (104, 170, 63): (64, 0, 128),  # motorcycle -> car
    (46, 227, 147): (0, 128, 192),  # bicycle -> bicycle
}

color_dict_CITYSCAPES_to_SYNTHIA_color_old = {
    (0, 0, 0): (0, 0, 0),  # void -> void
    (111, 74, 0): (0, 0, 0),  # dynamic -> void
    (81, 0, 81): (0, 0, 0),  # ground -> void
    (128, 64, 128): (128, 64, 128),  # road -> road
    (244, 35, 232): (0, 0, 192),  # sidewalk -> sidewalk
    (250, 170, 160): (128, 64, 128),  # parking -> road
    (230, 150, 140): (0, 0, 0),  # rail track -> void
    (70, 70, 70): (128, 0, 0),  # building -> building
    (102, 102, 156): (128, 0, 0),  # wall -> building
    (190, 153, 153): (64, 64, 128),  # fence -> fence
    (180, 165, 180): (64, 64, 128),  # guard rail -> fence
    (150, 100, 100): (128, 0, 0),  # bridge -> building
    (150, 120, 90): (128, 0, 0),  # tunnel -> building
    (153, 153, 153): (192, 192, 128),  # pole -> pole
    (250, 170, 30): (0, 128, 128),  # traffic light -> traffic light
    (220, 220, 0): (192, 128, 128),  # traffic sign -> traffic sign
    (107, 142, 35): (128, 128, 0),  # vegetation -> vegetation
    (152, 251, 152): (0, 0, 0),  # terrain -> void
    (70, 130, 180): (128, 128, 128),  # sky -> sky
    (220, 20, 60): (64, 64, 0),  # person -> pedestrian
    (255, 0, 0): (0, 128, 192),  # rider -> bicycle
    (0, 0, 142): (64, 0, 128),  # car -> car
    (0, 0, 70): (64, 0, 128),  # truck -> car
    (0, 60, 100): (64, 0, 128),  # bus -> car
    (0, 0, 90): (64, 0, 128),  # caravan -> car
    (0, 0, 110): (64, 0, 128),  # trailer -> car
    (0, 80, 100): (64, 0, 128),  # train -> car
    (0, 0, 230): (64, 0, 128),  # motorcycle -> car
    (119, 11, 32): (0, 128, 192),  # bicycle -> bicycle
    (0, 0, 142): (64, 0, 128)  # license plate -> car
}

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    img = Image.open(f'{base_path}{file}')
    out = cvt2color(np.array(img), color_dict=color_dict_CITYSCAPES_to_SYNTHIA_color)
    converted = Image.fromarray(out)
    converted.save(f'{out_path}{file}')
