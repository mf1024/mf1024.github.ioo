---
layout: post
title: "Leveraging unlabeled data with contrastive predictive coding"
---

As opposed to humans who can [generalize pretty well from few samples of new data](https://youtu.be/Ucp0TTmvqOE?t=7091) one of the biggest limitations of current state-of-the-art(at the time of writing this) deep learning models is that they require lots of labeled data to be trained to a level where they can become useful. Often labeled data is hard to collect, expensive to collect or even impossible to collect. But there are some techniques to deal with this problem to a certain degree.

![img1](/img/contrastive_predictive_coding/overview.png)

One of such techniques is semi-supervised learning. The idea of semi-supervised learning is making use of the unlabeled data as well as labeled data in the training.

In this post, I will present a semi-supervised method where feature extractor is trained using unlabeled data and then model that solves a specific task is trained on top of it using labeled data.

## Contrastive predictive coding

Paper [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) proposes an unsupervised learning approach for training feature extractor which can create useful and generic representations of the input data.

Extracting good feature vectors without supervision is a challenging task. Somehow the model without supervision needs to learn what is important in the high-dimensional input data and then it needs to compress the data into much less dimensional representation that contains as much important and useful information as the input.

CPC uses prediction as the objective to train the encoder to extract useful contextual representations.

Training a model to predict future or missing information is one of the most commonly used unsupervised strategies for creating good feature extractors. Authors of the paper hypothesize that this approach is successful because by solving the prediction problem the model must encode important high-level contextual information into the feature vector.

The model of CPC first encodes the data of current timestep into z-vector. Then it uses autoregressor to aggregate current z-vector and past z-vectors to create context vector. A context vector is then used to predict future z-vectors. For training negative sampling or contrastive loss is used which is similar to what is used for learning word embeddings in natural language models.


## The loss function of CPC

Objective set by CPC loss function is for the prediction $$ \hat{z}_{i+k} $$ to be most similar to the one positive sample $$ z_{i+k} $$ among a set of randomly selected negative samples $$ \{ z_l \} $$


$$
\mathcal{L}_{\mathrm{CPC}} =-\sum_{i, k} \log \frac{\exp \left(\hat{z}_{i+k}^{T} z_{i+k}\right)}{\exp \left(\hat{z}_{i+k}^{T} z_{i+k}\right)+\sum_{l} \exp \left(\hat{z}_{i+k}^{T} z_{l}^{\prime}\right)}
$$

$$ z_1^{T} z_2 $$ is a dot product between vectors which here is used as a vector similarity score.

$$ \frac{\exp \left(\hat{z}_{i+k, j}^{T} z_{i+k, j}\right)}{\exp \left(\hat{z}_{i+k, j}^{T} z_{i+k, j}\right)+\sum_{l} \exp \left(\hat{z}_{i+k, j}^{T} z_{l}^{\prime}\right)} $$ is softmax probability of the positive sample.

And $$ -\sum_{i, k} \log \ $$ sums the cross-entropy loss $$ -log(p) $$ over all time steps $i$ and offsets $k$  . 

<br>

## Making use of unlabeled data with CPC semi-supervised learning

Paper [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272) proposes to use CPC to train the feature extractor using unlabeled data and then to use a smaller amount of unlabeled data to train a classifier on top of the feature extractor. 

The intuition behind this method is that practically any deep learning model will spend first of its layers to extract useful features from the input of high-dimensional data. The latter layers will use those feature to solve its task. This semi-supervised method proposes to train generic feature extractor first without labeled data and then train a model for a specific task on top of it, which at this point is much easier because of feature extractor is doing some of the heavy lifting. 

## Proposed network architecture

CPC can be applied to different domains - speech, images, NLP, reinforcement learning. I will review CPC architecture applied for images.

### Unsupervised training for prediction

![img2](/img/contrastive_predictive_coding/cnn_architecuture.png)

In the case of images, the set objective for feature extractor training is to predict z-vectors of patches below the current patch.

First, the image is split into overlapping patches resulting into a grid of patches. Then each patch independently is encoded with encoder $$ g_{enc} $$ into feature vector resulting in a grid of feature vectors. To make the predictions context network is applied to the grid of feature vectors. In this case, the context vector is a deep convolutional network. Then the output of the context network  $$ c_t $$ is used to make the predictions. Given context vector $$ c_{i,j} $$ and offset $$ k $$ the prediction made linearly using a prediction matrix:

$$  \hat{z}_{i+k, j}=W_{k} c_{i, j} $$

We have different prediction matrix $$ W_{k} $$ for each offset $$ k $$. 

Now we can train the encoder network using unlabeled data.

### Supervised training for classification

![img3](/img/contrastive_predictive_coding/classifier.png)

Once the training for prediction is done we get rid of the context network(brown in the image above) and can now stack a classifier network(red) on top of the feature extractor(blue).

We now have a choice - either we can keep fine-tuning encoder params while we train the classifier network or we can freeze the encoder params and train only the classifier. Interestingly authors of the paper concluded that approach with freezing the encoder params yields almost the same performance as fine-tuning the params for the classification task.

Another interesting variation you could try in the classification training is to apply the encoder to the whole image instead of applying it to each of the patches.

Now we can train the classifier using labeled data.

## The Code

PyTorch Code is coming soon here.

