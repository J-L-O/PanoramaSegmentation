from os import listdir
from os.path import isfile, join

import click
import numpy as np
from shutil import copyfile


@click.command()
@click.option('--source', '-s', help='The folder containing the source images.', type=click.Path(exists=True))
@click.option('--destination', '-d', help='Where to store the cropped images.', type=click.Path(exists=True))
@click.option('--number', '-n', help='The number of images to select.', type=click.INT)
def split(source, destination, number):

    files = [f for f in listdir(source) if isfile(join(source, f))]

    np.random.seed(42)
    selected = np.random.choice(files, number, replace=False)

    for file in selected:
        copyfile(join(source, file), join(destination, file))


if __name__ == '__main__':
    split()
