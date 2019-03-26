# PyAutoLabeller

Autonomous robot for 2019 IEEE Region 5 Autonomous Robotics Competition.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Tested on Python 3.7.2

What things you need to install the software and how to install them

```
pip install pytorch
```

### Evaluate Training

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
The code to re-produce the model:
```
python eval_ssd.py --net vgg16-ssd --dataset vision/datasets/images/test/ --trained_model vision/models/vgg16-trained.pth --label_file vision/models/voc-model-labels.txt
```

## Authors

* **Rishav Rajendra** - [Website](https://rishavrajendra.github.io)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
