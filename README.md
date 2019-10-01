## [Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations](https://arxiv.org/abs/1811.08489)
[Tianlu Wang](http://www.cs.virginia.edu/~tw8cb/), [Jieyu Zhao](https://jyzhao.net/), [Mark Yatskar](http://markyatskar.com/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/), [Vicente Ordóñez](http://vicenteordonez.com/), ICCV 2019

### Abstract 
In this work, we present a framework to measure and mitigate intrinsic biases with respect to protected variables --such as gender--in visual recognition tasks. We show that trained models significantly amplify the association of target labels with gender beyond what one would expect from biased datasets. Surprisingly, we show that even when datasets are balanced such that each label co-occurs equally with each gender, learned models amplify the association between labels and gender, as much as if data had not been balanced! To mitigate this, we adopt an adversarial approach to remove unwanted features corresponding to protected variables from intermediate representations in a deep neural network -- and provide a detailed analysis of its effectiveness. Experiments on two datasets: the COCO dataset (objects), and the imSitu dataset (actions), show reductions in gender bias amplification while maintaining most of the accuracy of the original models.

### Requirements

### Data

### Demo instructions

### Training instructions

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
