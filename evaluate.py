import click as click
from keras_segmentation.predict import evaluate_from_files


@click.command()
@click.option('--predictions', '-p', help='The folder containing the predicted segmentation.', type=click.Path(exists=True))
@click.option('--ground_truth', '-gt', help='The folder containing the ground truth segmentation.', type=click.Path(exists=True))
@click.option('--n_classes', '-gt', default=16, help='The number of classes.', type=int)
def evaluate(predictions, ground_truth, n_classes):

    frequency_weighted_IU, mean_IU, cl_wise_score = evaluate_from_files(predictions, ground_truth, n_classes)

    print(f'frequency_weighted_IU: {frequency_weighted_IU}, mean_IU: {mean_IU}, class_wise_IU: {cl_wise_score}')


if __name__ == '__main__':
    evaluate()
