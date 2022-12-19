# Face embedding trainer

This repository contains the framework for training deep embeddings for face recognition. The trainer is intended for the face recognition exercise of the [EE488B Deep Learning for Visual Understanding](https://mm.kaist.ac.kr/teaching/) course. This is an adaptation of the [speaker recognition model trainer](https://github.com/clovaai/voxceleb_trainer).

[20180467 EE488B Experiment Report.pdf](https://github.com/seungyonglee0802/face_recognition/files/10259884/20180467.EE488B.Experiment.Report.pdf)

KSTAR Face Map
![tSNE_1](https://user-images.githubusercontent.com/74466088/208445283-528f83ad-783b-46ca-a1e7-debce16a0a4e.jpg)
![tSNE_2](https://user-images.githubusercontent.com/74466088/208445306-e7fccb4a-a8e1-476e-a27c-d8314b5e4cf5.jpg)

## GOALs

1. Train CNN model that can make appropriate embedding vector(nOut = 512) for korean star's faces. Evaluated by EER(equal error rate).
2. Get the most similar the korean star's face with a face with a random person(not famous)
3. Draw "KSTAR-FaceMap" which put faces nearby when they are similar each other, and vice versa.


### Dependencies
```
pip install -r requirements.txt
```

### Training examples

- Pretrain:
```
$ python ./trainEmbedNet.py --model ThinResNet50_V2 --train_path data/train/VGGFace2 
--trainfunc amsoftmax --scale 30 --margin 0.1 --save_path exps/T50V2 
--max_epoch 60 --nPerClass 8631 --max_img_per_cls 200 --batch_size 200 --lr 0.001 
--scheduler cosineRestartlr --gpu 0
```

GPU ID must be specified using `--gpu` flag.

Use `--mixedprec` flag to enable mixed precision training. This is recommended for Tesla V100, GeForce RTX 20 series or later models.

- Fine-tuning:
```
$ python ./trainEmbedNet.py --model ThinResNet50_V2 --initial_model exps/T50V2/model000000050.model 
--trainfunc angproto --nPerClass 2 --save_path exps/transfer_T50V2 
--max_epoch 50 --test_interval 1 --batch_size 250 --lr 0.0005 --scheduler cosineRestartlr --gpu 0
```

- Evaluation:
```
$ python ./trainEmbedNet.py --model ThinResNet50_V2 --initial_model exps/transfer_T50V2/model000000020.model
--trainfunc angproto --gpu 0 
--eval --test_path data/test_shuffle --test_list data/test_blind.csv --output output.csv
```

### Implemented loss functions
```
Softmax (softmax)
Triplet (triplet)
```

For softmax-based losses, `nPerClass` should be 1, and `nClasses` must be specified. For metric-based losses, `nPerClass` should be 2 or more. 

### Implemented models
```
ThinResNet50_V2
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
