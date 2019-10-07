## [Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations](https://arxiv.org/abs/1811.08489)
[Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Jieyu Zhao](https://jyzhao.net/), [Mark Yatskar](http://markyatskar.com/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/), [Vicente Ordóñez](http://vicenteordonez.com/), ICCV 2019

### Abstract 
In this work, we present a framework to measure and mitigate intrinsic biases with respect to protected variables --such as gender--in visual recognition tasks. We show that trained models significantly amplify the association of target labels with gender beyond what one would expect from biased datasets. Surprisingly, we show that even when datasets are balanced such that each label co-occurs equally with each gender, learned models amplify the association between labels and gender, as much as if data had not been balanced! To mitigate this, we adopt an adversarial approach to remove unwanted features corresponding to protected variables from intermediate representations in a deep neural network -- and provide a detailed analysis of its effectiveness. Experiments on two datasets: the COCO dataset (objects), and the imSitu dataset (actions), show reductions in gender bias amplification while maintaining most of the accuracy of the original models.

### Requirements
- Python 2.7
- Pytorch 0.4+
- Tensorflow (Tensorboard is used for logging)

### Data
* COCO
  1. Download images and annotations(2014) from [coco](http://cocodataset.org/#download)
  2. Create soft links in [data](./object_multilabel/data) folder:
  ```
  ln -s /path/to/annotations annotations_pytorch
  ln -s /path/to/train_images train2014
  ln -s /path/to/val_test_images val2014
  ```
* imSitu
  1. Download images from [imSitu](http://imsitu.org/download/)
  2. Create soft link in [data](./verb_classification/data) folder:
   ```
   ln -s /path/to/images of500_images_resized
   ```

### Demo instructions
* COCO
  * compute dataset_leakage(\lambda_D):
  ```
  CUDA_VISIBLE_DEVICES=5 python dataset_leakage.py --num_epochs 100 --learning_rate 0.00005 --batch_size 128 --num_rounds 5 --no_image
  ```
  * compute dataset leakage(\lambad_D(a)):
  ```
  CUDA_VISIBLE_DEVICES=5 python natural_leakage.py --num_rounds 5 --num_epochs 100 --learning_rate 0.00005 --batch_size 128 --no_image
  ```
  * compute model leakage:
  ```
  CUDA_VISIBLE_DEVICES=1 python attacker.py --exp_id origin_0 --num_rounds 5 --num_epochs 100 --learning_rate 0.00005 --batch_size 128
  ```
  
  * visualize debiased images:
  ```
  cd adv
  python  vis.py --exp_id path_to_checkpoint
  cd sample_images
  python -m SimpleHTTPServer
  ```
  
  

### Training instructions
For more detailed commands, please refer to commands.ipynb
* COCO
  * run train.py (adv/train.py or adv/ae_adv_train.py) to train a regular (adversarial or encoder+adversarial) model, e.g.:
  ```
  CUDA_VISIBLE_DEVICES=1 python train.py --save_dir origin_0 --log_dir origin_0 --batch_size 32 --num_epochs 60 --learning_rate 0.0001
  ```
  * to track the training procedure, go to ./logs and run:
  ```
  tensorboard --logdir .
### Citing
If you find our paper/code useful, please consider citing:

```
@InProceedings{wang2019iccv,
author={Tianlu Wang and Jieyu Zhao and Mark Yatskar and Kai-Wei Chang and Vicente Ordonez},
title={Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations},
booktitle = {International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
