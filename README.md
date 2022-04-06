# ultramnist
my code for [ultra mnist](https://www.kaggle.com/c/ultra-mnist) challenge on kaggle. Training-Val plots [here](https://wandb.ai/harshraj22/ultramnist?workspace=user-harshraj22)

## Note:
It does not work yet, and now I am too lazy to debug. 

## Innovation:
Tried to pretrain model on task to predict rotation, flipping. The idea was to teach the model to understand what to look for in the images (the digits in this case). 4000px image was cropped into 4 1000px images, and one sample of dataset contained 4 images. The pretained model would give a good set of weights for feature extraction trained on a simple task. We would then use this network for the more difficult task of classification of digits sum. See pretrain directory for more details.


### Directory Structure:
```bash
.
├── conf
│   └── config.yaml                <- All config files
├── data_loader
│   └── data_loader.py
├── explain                        <- Scripts for Model explainability
│   ├── Readme.md
│   ├── exp3
│   │   ├── input.jpg
│   │   ├── result1.jpg
│   │   └── result2.jpg
│   └── visualize.py
├── models                        <- All Model files
│   └── mobilenetv3.py
├── requirements.txt
├── rough                         <- Rough scripts for experimentation
│   └── weight_load.py
├── train.py                      <- Train the model
├── utils
│   └── utils.py
└── weights                       <- Model weights
    ├── model_weight.pth
    └── model_weight.pth.dvc
```


## Checks for Training from scratch:
- [x] Overfits one batch
- [x] All weights are getting changed
- [x] Didn't forget to call `optimizer.zero_grad( )`
- [x] No Mixing of training and testing data

