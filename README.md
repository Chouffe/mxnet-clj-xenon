# xenon

## Network Architecture and Training

> We used a 169-layer convolutional neural network to predict the probability of abnormality for each image in a study. The network uses a Dense Convolutional Network architecture – detailed in Huang et al. (2016) – which connects each layer to every other layer in a feed-forward fashion to make the optimization of deep networks tractable.  We replaced the final fully connected layer with one that has a single output, after which we applied a sigmoid nonlinearity.
> For each image $$X$$ of study type $$T$$ in the training set, we optimized the weighted binary cross entropy loss.
> Before feeding images into the network,  we normalized each image to have the same mean and standard deviation of images in the ImageNet training set. We then scaled the variable-sized images to $$320 × 320$$.  We augmented the data during training by applying random lateral inversions and rotations of up to 30 degrees.
> The weights of the network were initialized with weights from a model pretrained on ImageNet (Deng et al., 2009). The network was trained end-to-end using Adam with default parameters $$\beta_1 = 0.9$$ and $$\beta_2 = 0.999$$ (Kingma & Ba, 2014). We trained the model using minibatches of size $$8$$. We used an initial learning rate of $$0.0001$$ that is decayed by a factor of $$10$$ each time the validation loss plateaus after an epoch. We ensembled the $$5$$ models with the lowest validation losses.

From above 3 paragraphs to reproduce this we need:

1. Model
    - [x] DenseNet169 as base (as well as few other popular image recognition models like other DenseNet variations or different ResNet)
    - [x] Pretrained on ImageNet
    - [x] Last layer changed to binary classification
    - [x] cross entropy loss function
    - [ ] weighted cross entropy loss
2. Prepare data
    - [X] MURA dataset
    - [x] Resized beforehand for training
    - [x] Set data to proper directory structure
    - [x] Data augmentation (implemented with `RecordIO`)
3.  Training
    - [x] Init with pretrained weights
    - [x] Adam optimizer with gicen hyperparameters
    - [x] Training hyperparameters
    - [ ] Decaying learnign rate after validation loss plateaus
    - [x] Ensemble 5 bests

## resources

### MURA

> MURA (musculoskeletal radiographs) is a large dataset of bone X-rays. Algorithms are tasked with determining whether an X-ray study is normal or abnormal.

* [Paper](https://arxiv.org/abs/1712.06957)
* [MURA Dataset](https://stanfordmlgroup.github.io/competitions/mura/)

### DenseNet

> Dense Convolutional Network (DenseNet) connects each layer to every other layer in a feed-forward fashion.

* [Paper](https://arxiv.org/abs/1608.06993)
* [MXNet implementation](https://github.com/miraclewkf/DenseNet)

## dev

### im2rec

* [incubator-mxnet/tools/im2rec](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py)
* [prepare datasets example](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification#prepare-datasets)

### docker

Command to run docker with exposed REPL:
```bash
nvidia-docker run --name xenon-repl -p 3133 -v <absolute-path>/xenon/app:/home/magnet/app -it xenon lein repl :headless :host 0.0.0.0 :port 3133
```

Find container IP
```bash
docker inspect xenon-repl | grep IPAddress
```
