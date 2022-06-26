# Multi class lightGBM with focal loss

Confusion matrix on the test set using the standard LightGBM classifier

![Confusion matrix on the test set using the LGBMClassifier](https://github.com/lucacarniato/multi-class-lightgbm-with-focal-loss/blob/main/figures/LightGBMConfusionMatrix.png?raw=true)

Confusion matrix on the test set using LightGBM and the customized multi-class Focal Loss class (OneVsRestLightGBMWithCustomizedLoss)

![Confusion matrix on the test set using the OneVsRestLightGBMWithCustomizedLoss](https://github.com/lucacarniato/multi-class-lightgbm-with-focal-loss/blob/main/figures/LightGBMFocalLossConfusionMatrix.png?raw=true)

## Introduction

This repository contains the source code of the medium post [Multi-Class classification using Focal Loss and LightGBM](https://towardsdatascience.com/multi-class-classification-using-focal-loss-and-lightgbm-a6a6dec28872)

In post details how focal loss can be used for a multi class classification LightGBM model.

By using Focal Loss, sample weight balancing, or artificial addition of new samples to reduce the imbalance are not required. 
On an artificially generated multi-class imbalanced dataset, the use of Focal loss increased the recall value and eliminated some false positives and negatives in the minority classes.

