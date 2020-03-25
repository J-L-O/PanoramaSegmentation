import click as click
from PIL import Image
from keras_segmentation.data_utils.data_loader import get_pairs_from_paths
from tqdm import tqdm
import numpy as np


@click.command()
@click.option('--predictions', '-p', help='The folder containing the predicted segmentation.', type=click.Path(exists=True))
@click.option('--ground_truth', '-gt', help='The folder containing the ground truth segmentation.', type=click.Path(exists=True))
@click.option('--n_classes', '-gt', default=16, help='The number of classes.', type=int)
def evaluate(predictions, ground_truth, n_classes):

    paths = get_pairs_from_paths(ground_truth, predictions)
    paths = list(zip(*paths))
    inp_images = list(paths[0])
    annotations = list(paths[1])

    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    n_pixels = np.zeros(n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = np.array(Image.open(inp))
        gt = np.array(Image.open(ann))

        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(n_classes):
            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * (gt != cl_i))
            fn[cl_i] += np.sum((pr != cl_i) * (gt == cl_i))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    print(f'frequency_weighted_IU: {frequency_weighted_IU}, mean_IU: {mean_IU}, class_wise_IU: {cl_wise_score}')


if __name__ == '__main__':
    evaluate()
