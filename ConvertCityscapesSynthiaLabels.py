from os import listdir
from os.path import isfile, join
from PIL import Image

from convert import convert_colors

# input = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'
# output = './SYNTHIA-PANO/PREDICTIONS_Converted_Labels/seqs02_fall/'

base_path = './panoramabilder/Langenbeckerstr_Predictions/Pretrained_split/'
out_path = './panoramabilder/Langenbeckerstr_Predictions/Pretrained_split_Colors/'

color_dict_CITYSCAPES_to_SYNTHIA_labels = {
    (20, 215, 197): 3,  # road -> road
    (207, 248, 132): 4,  # sidewalk -> sidewalk
    (183, 244, 155): 2,  # building -> building
    (144, 71, 111): 2,  # wall -> building
    (128, 48, 71): 5,  # fence -> fence
    (50, 158, 75): 7,  # pole -> pole
    (241, 169, 37): 15,  # traffic light -> traffic light
    (222, 181, 51): 9,  # traffic sign -> traffic sign
    (244, 104, 161): 6,  # vegetation -> vegetation
    (31, 133, 226): 0,  # terrain -> void
    (204, 47, 7): 1,  # sky -> sky
    (170, 252, 0): 10,  # person -> pedestrian
    (32, 166, 124): 11,  # rider -> bicycle
    (122, 113, 97): 8,  # car -> car
    (46, 229, 72): 8,  # truck -> car
    (250, 163, 41): 8,  # bus -> car
    (149, 154, 55): 8,  # train -> car
    (104, 170, 63): 8,  # motorcycle -> car
    (46, 227, 147): 11,  # bicycle -> bicycle
}

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    img = Image.open(f'{base_path}{file}')
    converted = convert_colors(img, color_dict_CITYSCAPES_to_SYNTHIA_labels, False)
    converted.save(f'{out_path}{file}')
