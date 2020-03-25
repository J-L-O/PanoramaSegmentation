from os import listdir
from os.path import isfile, join

import click
from PIL import Image, ImageFilter


@click.command()
@click.option('--input', '-i', help='The folder containing the input images.', type=click.Path(exists=True))
@click.option('--output', '-o', help='Where to store the converted images.', type=click.Path(exists=True))
def remove_small_objects(input, output):
    files = [f for f in listdir(input) if isfile(join(input, f))]

    for file in files:
        label = Image.open(f'{input}{file}')
        blurred_label = label.filter(ImageFilter.ModeFilter(5))
        blurred_label = blurred_label.filter(ImageFilter.ModeFilter(5))
        blurred_label = blurred_label.filter(ImageFilter.ModeFilter(5))
        blurred_label = blurred_label.filter(ImageFilter.ModeFilter(5))
        blurred_label.save(f'{output}{file}')


if __name__ == '__main__':
    remove_small_objects()

