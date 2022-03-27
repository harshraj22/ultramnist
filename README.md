# ultramnist
my code for [ultra mnist](https://www.kaggle.com/c/ultra-mnist) challenge on kaggle



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

