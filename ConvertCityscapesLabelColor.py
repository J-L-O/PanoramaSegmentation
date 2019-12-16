from convert import convert_colors

base_path = './SYNTHIA-PANO/PREDICTIONS/seqs02_fall/'
out_path = './SYNTHIA-PANO/PREDICTIONS_Converted_Labels/seqs02_fall/'

color_dict_CITYSCAPES_to_Color = {
    0: (20, 215, 197),  # road
    1: (207, 248, 132),  # sidewalk
    2: (183, 244, 155),  # building
    3: (144, 71, 111),  # wall
    4: (128, 48, 71),  # fence
    5: (50, 158, 75),  # pole
    6: (241, 169, 37),  # traffic light
    7: (222, 181, 51),  # traffic sign
    8: (244, 104, 161),  # vegetation
    9: (31, 133, 226),  # terrain
    10: (204, 47, 7),  # sky
    11: (170, 252, 0),  # person
    12: (32, 166, 124),  # rider
    13: (122, 113, 97),  # car
    14: (46, 229, 72),  # truck
    15: (250, 163, 41),  # bus
    16: (149, 154, 55),  # train
    17: (104, 170, 63),  # motorcycle
    18: (46, 227, 147),  # bicycle
}


def convert_image(image):
    return convert_colors(image, color_dict_CITYSCAPES_to_Color, True)
