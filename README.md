Code for "Differences between hard and noisy-labeled samples: An empirical study".

## Requirements
We conducted experiments under:
- python 3.7.10
- torch 1.7.1
- torchvision 0.8.2
- cuda 10.1
- jupyter-notebook 6.1.5
- ipython 7.19.0
- 1 Nvidia Titan X Maxwell GPU


## Description of directories
* Imbalance directory: codes and preprocessing related to experiments of hardness via imbalance.
* Diversification directory: codes and preprocessing related to experiments of hardness via diversification.
* Closeness directory: codes and preprocessing related to experiments of hardness via closeness to the decision boundary.

## Description of files in each directory
In each directory you will find the following files:
* dataprocessing.py: the function that produces the transformed version of the TinyImageneNet dataset with custom hardness and noisiness levels per samples of different classes.
* train.ipynb: code that trains models on datasets with hard and noisy samples.

## A Walk-through
After training a model on the dataset with hard and noisy samples and saving the values in a checkpoint, you could use the data_partitioning.ipynb to partition the dataset.

## Access to the paper

You can find the full version of the paper (including appendices) at https://arxiv.org/abs/2307.10718.


## Citation

To cite our work, for the arxiv version, use:
```
@article{
}
```
