# HybridNets implementation


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#file-structure">File Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installing-requirements">Installing requirements</a></li>
        <li><a href="#download-and-unzip-prepared-dataset">Download and unzip prepared dataset</a></li>
        <li><a href="#download-pretrained-checkpoint">Download pretrained checkpoint</a></li>
        <li><a href="#download-and-unzip-prepared-dataset">Download and unzip prepared dataset</a></li>
        <li><a href="#train-the-model">Train the model</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#generate-images-of-losses">Generate images of losses</a></li>
        <li><a href="#generate-sample-images">Generate sample images</a></li>
      </ul>
    </li>
  </ol>
</details>


## About
This is a sample implementation of the original paper: HybridNets end-to-end perception network.
Created by Bence Kátai-Pál (bence.kataipal [at] gmail.com)

> [**HybridNets: End-to-End Perception Network**](https://arxiv.org/abs/2203.09035)
>
> by Dat Vu, Bao Ngo, Hung Phan [*FPT University*](https://uni.fpt.edu.vn/en-US/Default.aspx)
>
> *arXiv technical report ([arXiv 2203.09035](https://arxiv.org/abs/2203.09035))*

HybridNets is a multi-task model for autonomous driving tasks, like traffic object detection, drivable area segmentation and lane line detection. The model became real-time state-of-the-art on BDD100K dataset on traffic object detection and lane line detection.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybridnets-end-to-end-perception-network-1/traffic-object-detection-on-bdd100k)](https://paperswithcode.com/sota/traffic-object-detection-on-bdd100k?p=hybridnets-end-to-end-perception-network-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybridnets-end-to-end-perception-network-1/lane-detection-on-bdd100k)](https://paperswithcode.com/sota/lane-detection-on-bdd100k?p=hybridnets-end-to-end-perception-network-1)


### File Structure
```bash
HybridNets
│   train.py                      # Train script
│   val.py                        # Validator script
│   results.py                    # Loss plotter
│   samples.py                    # Sample image plotter
│   requirements.txt              # Package dependency file
│
│
└───hybridnets
        backbone.py            	  # Backbone modul
        dataset.py                # BDD100K dataset generator
        detection_head.py         # Detection head
        loss.py                   # Loss handler script
        neck.py                   # Neck modul
        network_block.py          # Network blocks and layers
        segmentation_head.py      # Segmentation head
        smp_metrics.py            # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/metrics/functional.py
        hybridnets.yml            # Parameter file
```

## Getting Started [![Open In Colab](https://colab.research.google.com/drive/1cbj5R19PVsfHvhinLMDolEdmFIJv6i5A)
For running the model it is necessary to use GPU. Please set the environment in Colab.


### Installing requirements
To be able to clone the repository please make sure that the following packages are installed
```bash
apt-get install git unzip wget -y
```


Clone the repository and install dependencies
```bash
git clone https://github.com/k3wj8h/HybridNets.git
cd HybridNets

pip install -r requirements.txt
```


### Download and unzip prepared dataset
The database is the 100K Image database of BDD100K with a simplified structure of labels
```bash
gdown 1sj0lC4IPluN8armUoK4iw6Wq5jIQ0l4F
unzip -q -o datasets.zip
``` 

Recommended dataset structure:
```bash
HybridNets
└───datasets
    ├───bdd100k
    │   ├───train
    │   └───val
    ├───det_annotations
    │   ├───train
    │   └───val
    ├───da_seg_annotations
    │   ├───train
    │   └───val
    └───ll_seg_annotations
        ├───train
        └───val
```

Original dataset is available at [BDD100K](https://bdd-data.berkeley.edu/)



### Download pretrained checkpoint
```bash
mkdir checkpoint
cd /content/HybridNets/checkpoint
#gdown 1TUPwmgZ9UV3TZwafyhQ6NRyI8-6hwGgn
cd /content/HybridNets
```


### Train the model
```bash
python train.py --num_epochs        # Number of epochs (default=5)
                --num_gpus          # Number of GPUs (default=1)
                --num_sample_images # Number of sample images (default=3)
                --load_checkpoint   # Path of previously saved checkpoint, set None to initialize
                --param_file        # Path of parameter file (default='./hybridnets/hybridnets.yml')
```
Please check `python train.py --help` for available arguments.

Checkpoints and sample images will be saved automatically at the end of the epochs.
Folder of checkpoints: *checkpoint*
Folder of sample images: *sample_images*


### Evaluation
```bash
python val.py --batch_size      # Number of images in a batch (default=10)
              --num_gpus        # Number of GPUs to be used (default=1)
              --checkpoint      # Path of previously saved checkpoint
              --param_file      # Path of parameter file (default='./hybridnets/hybridnets.yml')
```
Please check `python val.py --help` for available arguments.


### Generate images of losses
```bash
python results.py --checkpoint          # Path of previously saved checkpoint
                  --image_path          # Path to save images (default='.sample_imgages')
```
Please check `python results.py --help` for available arguments.
Images will be saved into *sample_images* folder


### Generate sample images
```bash
python samples.py --checkpoint          # Path of previously saved checkpoint
                  --image_path          # Path to save images (default='.sample_imgages')
                  --num_sample_images   # Number of sample images (default=1)
                  --param_file          # Path of parameter file (default='./hybridnets/hybridnets.yml')
                  --dataset             # Dataset train or val
```
Please check `python val.py --help` for available arguments.
Sample images will be saved into *sample_images* folder

