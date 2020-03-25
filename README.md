# PanoramaSegmentation

Contains commands to train and evaluate segmentation models on panoramic images. Install the requirements with `pip install -r requirements.txt` before you get started.

Usage example:

First, train the model with

`python .\PanoramaEvaluation\train.py --images .\Data\RGB\train_5\ --labels .\Data\LABELS\train_5\ --model resnet50_pspnet --width 1536 --height 384 --name _5`

Then predict with

`python .\PanoramaEvaluation\predict.py --images .\Data\RGB\test\ --predictions .\Data\resnet50_pspnet_1536x384_5\test\ --from-checkpoint --checkpoint_dir .\checkpoints\resnet50_pspnet_1536x384_5\resnet50_pspnet_1536x384_5`

Convert the RGB segmentation into grayscale so it can be evaluated

`python .\PanoramaEvaluation\convert.py --input .\Data\resnet50_pspnet_1536x384_5\test\ --output .\Data\resnet50_pspnet_1536x384_5\test_label\ --conversion DEFAULT_Color_to_SYNTHIA_Label`

Run the evaluation script

`python .\PanoramaEvaluation\evaluate.py --predictions .\Data\resnet50_pspnet_1536x384_5\test_label\ --ground_truth .\Data\LABELS\test\ --n_classes 16`