# Face embedding trainer

This repository contains the framework for training deep embeddings for face recognition. The trainer is intended for the face recognition exercise of the [EE488B Deep Learning for Visual Understanding](https://mm.kaist.ac.kr/teaching/) course. This is an adaptation of the [speaker recognition model trainer](https://github.com/clovaai/voxceleb_trainer).

## GOALs

1. Train CNN model that can make appropriate embedding vector(nOut = 512) for korean star's faces. Evaluated by EER(equal error rate).
2. Get the most similar the korean star's face with a face with a random person(not famous)
3. Draw "KSTAR-FaceMap" which put faces nearby when they are similar each other, and vice versa.

### Strategy

(Pretrain)
- DataSet: VGGFace2(8631 classes, max_images_per_class = 100 to overcome class imbalance)
- Model: ThinResNet34(~5M param)
- Loss: AMsoftmax(m=0.1, s=30)
- lr/schedule: start with lr = 0.001 make x0.9 after every 5 steps 
- Epoch: 70
- Batch Size 200
- Validation EER: 
- Data Augmentation: random crop 0.8

### Dependencies
```
pip install -r requirements.txt
```

### Training examples

- Softmax:
```
python ./trainEmbedNet.py --model ResNet18 --trainfunc softmax --save_path exps/exp1 --nClasses 2000 --batch_size 200 --gpu 8
```

GPU ID must be specified using `--gpu` flag.

Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

### Implemented loss functions
```
Softmax (softmax)
Triplet (triplet)
```

For softmax-based losses, `nPerClass` should be 1, and `nClasses` must be specified. For metric-based losses, `nPerClass` should be 2 or more. 

### Implemented models
```
ResNet18
```

### Adding new models and loss functions

You can add new models and loss functions to `models` and `loss` directories respectively. See the existing definitions for examples.

### Data

The test list should contain labels and image pairs, one line per pair, as follows. `1` is a target and `0` is an imposter.
```
1,id10001/00001.jpg,id10001/00002.jpg
0,id10001/00003.jpg,id10002/00001.jpg
```

The folders in the training set should contain images for each identity (i.e. `identity/image.jpg`).

The input transformations can be changed in the code.

### Inference

In order to save pairwise similarity scores to file, use `--output` flag.
