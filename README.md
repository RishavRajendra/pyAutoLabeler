# PyAutoLabeler

PyAutoLabeler is a simple tool that helps accelerate the image labeling process for custom mobile detection models. The main idea behind this project is to train a slower but very accurate model with the available labeled dataset and use the trained model to label additional images for the faster but not so accurate model. Follows the XML format created by [tzutalin/labelImg](https://github.com/tzutalin/labelImg). Newly labelled images can be validated using LabelImg.

Single Shot Multibox Detector(SSD) implementation in PyTorch burrowed from:

* [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)


## Preparation

First, clone this repository:

```
https://github.com/RishavRajendra/pyAutoLabeler.git
cd pyAutoLabeler
```

### Prerequisites

Tested on Python 3.7.1

What things you need to install the software and how to install them

```
pip3 install torch torchvision
pip3 install opencv-python
pip3 install pandas
```

### Data Preparation

We are following the Pascal VOC dataset format. So the images and annotations need to be in the following structure:

```
|-pytorch
|-datasets
    |-images
        |-test
            |-Annotations
            |-JPEGImages
            |-test.txt
        |-train
            |-Annotations
            |-JPEGImages
            |-train.txt
```

Populate test.txt and train.txt with the names of the images in test and train folders respectively. For ex: ```image01.jpg``` should be ```image01```. I recommend using ```ls Annotations/*.xml > train.txt``` and removing the ```.xml``` using a text editor.

Replace the class names in ```pytorch/voc_dataset.py``` and ```pytorch/models/voc-model-labels.txt``` with your own class labels.

### Pre-trained Model

Download pre-trained models:

```
wget -P models https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
wget -P models https://storage.googleapis.com/models-hao/mb2-imagenet-71_8.pth
```

### Train

Train the VGG based SSD implementation:

```
cd pytorch
python train_ssd.py --train_dataset datasets/images/train/ --test_dataset datasets/images/test/ --net vgg16-ssd --base_net models/vgg16_reducedfc.pth  --batch_size 24 --num_epochs 200 --scheduler "multi-step” —-milestones “120,160”
```

### Evaluate Training

Highest precision per-class I achieved:

```
Average Precision Per-class:
slope: 0.8559686187901899
start: 0.8570291777188328
blocka: 0.9020976917186305
blockb: 0.869311086160904
blockc: 0.8974831184775937
blockd: 0.879126840705184
blocke: 0.8879779277601088
blockf: 0.8877579605029737
obstacle: 0.8977965373066247
side: 0.7882676643651874
corner: 0.8587204123994634

Average Precision Across All Classes:0.8710488214459722
```
Code to evaluate the model:
```
python eval_ssd.py --net vgg16-ssd --dataset vision/datasets/images/test/ --trained_model vision/models/vgg16-trained.pth --label_file vision/models/voc-model-labels.txt
```

## Label new images

Label new images using the newly trained model:

```
python img_to_xml.py <path_to_images> <path_to_annotations> <model_path> <label_path>
```

## Tensorflow Support

If you want to use the labelled images for Tensorflow Object Detection API, convert XML to CSV:

```
python xml_to_csv.py <dataset_path>
```
Dataset should follow the directory structure shown in Data Preparation.

## Authors

* **Rishav Rajendra** - [Website](https://rishavrajendra.github.io)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
