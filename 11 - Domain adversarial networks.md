# Domain adversarial networks (DANs) also known as Unsupervised domain adaptation (UDA)

## Introduction

Domain adversarial networks (DANs) are a type of neural network architecture designed to address the problem of domain shift, where the distribution of data differs between the training and testing phases. DANs use adversarial training to learn domain-invariant features that are robust across different domains.

## Aim
The aim is to learn a classifier that performs well on both the source and target domains, even though the target domain may have different characteristics than the source domain. For example, we train a model to classify medical images at company A and then we want to use this model to classify medical images at some other hospital.

## Problem setting
Given:
- A source domain $D_s$ = {$(x_i^s, y_i^s)$}, $i = 1, ..., n$ with data distribution $p_s(x, y)$
- A target domain $D_t$ = {$(x_i^t)$}, $i = 1, ..., j$ with data distribution $p_t(x)$

Note that we don't have access to the target labels $y_i^t$ for the target domain.

Goal: learn a classifier $h(x)$ that performs well on the target domain, even though the target domain may have different characteristics than the source domain.

## Approach

### Feature extractor - Domain invariant feature learning
First a feature extractor is learned that is able to extract features that are domain invariant.

<div style="text-align: center;"><img src="19.jpg" alt="Image Description" width="400" height="auto"/></div>

Here $Z_s = \phi(x^s)$ and $Z_t = \phi(x^t)$ are the features extracted from the source and target domains respectively.

### Domain discriminator

Let $P_{Z_s}$ and $P_{Z_t}$ denote their distributions of $Z_s$ and $Z_t$ respectively. Our goal is to:

$$ \phi^* = \arg\min_{\phi} D(P_{Z_s} \| P_{Z_t}) $$
We can use adversarial training to learn this feature extractor. We introduce a domain discriminator $D(z)$ that attempts to distinguish between the source and target features.
<div style="text-align: center;"><img src="18.jpg" alt="Image Description" width="400" height="auto"/></div>

### Training of feature extractor and domain discriminator
The domain adaptation loss is given by:
$$ \mathcal{L}_{DA} = \mathbb{E}_{x^s \sim p_s(x)}[\log(D(\phi(x^s)))] + \mathbb{E}_{x^t \sim p_t(x)}[\log(1 - D(\phi(x^t)))] $$

The feature extractor is trained to minimize this loss, while the domain discriminator is trained to maximize it.

### Classifier

We train a classifier using pairs of features and labels from the source domain, i.e, $D_s$ = {$(Z_i^s, y_i^s)$}, i = 1, ..., n, where $Z_i^s = \phi(x_i^s)$ is the extracted feature corresponding to each source data point. This trained classifier is then used to predict the labels of the target domain data.

In actual practice:
- The classifier is trained jointly with the feature extractor and domain discriminator in an end-to-end manner
<div style="text-align: center;"><img src="20.jpg" alt="Image Description" width="600" height="auto"/></div>

- The total loss function combines:
  1. The classification loss on source domain data
  2. The domain adaptation loss between source and target domains
- The feature extractor is trained to:
  1. Minimize the classification loss (to learn discriminative features)
  2. Minimize the domain adaptation loss (to learn domain-invariant features)
- The domain discriminator is trained to maximize the domain adaptation loss

## Takeaway
- Whenever you encounter a problem where you have a source domain and a target domain, you should first try to learn a domain invariant feature extractor.
- Adversarial training is a powerful technique that should come to your mind whenever you encounter a problem where you want to minimize the distance between two distributions.






