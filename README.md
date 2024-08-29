# StarDiffusion [![arXiv](https://arxiv.org/static/arxiv_logo.png)]([https://arxiv.org](https://arxiv.org/abs/2211.09206)) [![Hugging Face](https://huggingface.co/front/assets/huggingface_logo.svg)](https://huggingface.co/datasets/pandaphd/Star_Field_Image_Enhancement_Benchmark)


Offical coda and dataset for [Learning to Kindle the Starlight][https://arxiv.org/abs/2211.09206]

## Structure of this project
```
├─checkpoint  // the weight files generated during the training process
├─weights  // the weights we provide for quick inference 
├─test_result // Storage of test results generated during testing
├─args_file.py  // Set the parameters needed for training
├─requirements.txt // The packages needed for this project
├─inference.py       // Inference Scripts
└─train.py      // Scripts for  training
```

## Dataset
we construct the first Star Field Image Enhancement Benchmark (SFIEB) that contains 355 real-shot and 854 semi-synthetic star field images, all having the corresponding reference images. You can download the dataset from [Hugging Face][https://huggingface.co/datasets/pandaphd/Star_Field_Image_Enhancement_Benchmark]. Each image has a resolution of 640*640.


## Usage
Before using this project, be sure to review the project structure above
### Train
First, open **args_file.py** to set the parameters needed for training, the run **train.py**
The weight files generated during training are saved in the **checkpoint** file, and the tensorboard files are saved in the **logs** folder

### Inference
Run inference.py, then set the following parameters and run test.py to test the test data set


## Reference
```
@ARTICLE{Yu2022,
  author={Yuan, Yu and Wu, Jiaqi and Wang, Lindong and Jing, Zhongliang ang Leung, Henry and Pan, Han},
  journal={arXiv}, 
  title={Learning to Kindle the Starlight}, 
  year={2022},
}
```
