from os import listdir
from os.path import isfile, join
from PIL import Image

from convert import convert_colors

# input = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'
# output = './SYNTHIA-PANO/PREDICTIONS_Converted_Labels/seqs02_fall/'

base_path = './panoramabilder/Langenbeckerstr_Predictions/Trained_473x473/'
out_path = './panoramabilder/Langenbeckerstr_Predictions/Trained_473x473_Labels/'

color_dict_SYNTHIA_color_to_Label = {
    (20, 215, 197): 0,  # void
    (207, 248, 132): 1,  # sky
    (183, 244, 155): 2,  # building
    (144, 71, 111): 3,  # road
    (128, 48, 71): 4,  # sidewalk
    (50, 158, 75): 5,   # fence
    (241, 169, 37): 6,  # vegetation
    (222, 181, 51): 7,   # pole
    (244, 104, 161): 8,  # car
    (31, 133, 226): 9,  # traffic sign
    (204, 47, 7): 10,  # pedestrian
    (170, 252, 0): 11,  # bicycle
    (32, 166, 124): 12,  # landmarking
    (122, 113, 97): 13,  # void
    (46, 229, 72): 14,  # void
    (250, 163, 41): 15  # traffic light
}

files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

for file in files:
    img = Image.open(f'{base_path}{file}')
    converted = convert_colors(img, color_dict_SYNTHIA_color_to_Label, False)
    converted.save(f'{out_path}{file}')
