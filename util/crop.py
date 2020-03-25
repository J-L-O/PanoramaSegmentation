from os import listdir
from os.path import isfile, join

import click
from PIL import Image


@click.command()
@click.option('--source', '-s', help='The folder containing the source images.', type=click.Path(exists=True))
@click.option('--destination', '-d', help='Where to store the cropped images.', type=click.Path(exists=True))
def crop(source, destination):

    files = [f for f in listdir(source) if isfile(join(source, f))]

    for file in files:
        image = Image.open(join(source, file))
        cropped = image.crop((0, 0, 8000, 3000))
        cropped.save(join(destination, file))


if __name__ == '__main__':
    crop()
