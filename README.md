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

## Description of files in each directory:
In each directory you will find the following files:
* dataprocessing.py: the function that produces the transformed version of the TinyImageneNet dataset with custom hardness and noisiness levels per samples of different classes.
* 


## Example to train
To train a resnet on the CIFAR-10 dataset with 50% label noise level, batch size=128, for 200 epochs run the following command:

```
python3 experiments.py --model resnet --filename <filename> --modelfilename <modelfilename>
```

## A walk-through on



## Access to the paper

You can find the full version of the paper (including appendices) at https:


## Citation

To cite our work, for the arxiv version, use:
```
@article{
}
```
