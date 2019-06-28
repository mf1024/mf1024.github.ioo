---
layout: post
title: "Notes on EfficientNets"
comments: true
---
> **notes on [https://arxiv.org/abs/1905.11946 ](https://arxiv.org/abs/1905.11946)
## Key contributions and ideas of this paper:
- Authors of the paper study width, depth, and resolution scaling effects on the accuracy of a ConvNet
- They find that it is critical to balance the scaling of all 3 dimensions and they introduce the **compound scaling method**
- They apply RL search proposed by [MnasNet paper](https://arxiv.org/abs/1807.11626) to find the baseline **EfficientNet-B0**
- They come up with **EfficientNet family** that reaches new state-of-the-art accuracy by using **10x** fewer parameters


Authors of the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) presents a systematic method for finding efficient convolutional network architectures. Using this method, they obtained a family of networks called **EfficientNets**. 

**EfficientNets achieve dramatically better efficiency than previous ConvNets,** including ResNets, DenseNets, Inception. 
![accuracy](/img/efficientnet/accuracy.png)
**EfficientNet achieve the same accuracy on ImageNet as other ConvNets by using up to 8.4x fewer parameters and up to 16x fewer FLOPS.** EfficientNet-B7 even achieves state-of-the-art 84.4% top-1/ 97.1% top-5 accuracy on ImageNet, while using 8.4x fewer parameters, and being 6.1x faster than the best existing ConvNet -  GPipe. 


## The goal

**In this paper, they investigate whether there is a principled method to scale up the Convolutional networks that can achieve better accuracy and efficiency.**

There are many ways to scale up a ConvNet. To simplify the scaling problem, first they fix some baseline ConvNet architecture (see the image below) and scale each network stage in three dimensions:
- width - number of channels
- depth - number of layers
- resolution - height times width of the tensor

**They define an optimization problem which they study: how to scale up the network in width, depth, and resolution such that the network fits within given memory and FLOPS constraints and gives the best performance.**

![scaling](/img/efficientnet/scaling.png)

## ConvNet scaling

In terms of efficiency, scaling up ConvNets is not well-understood process because the optimal design search space is large, and the tuning process is usually slow, tedious, and expensive. 

Most commonly, networks are scaled in depth by adding more layers, but as they show in their study **scaling up in just one dimension is not an optimal use of additional resources assigned to the ConvNet**. 

Experimentally they find that **it is critical to balance ConvNet scaling in all three of the dimensions** width, depth, and height to achieve better performance and efficiency. 

##  Compound scaling method

They propose a simple yet powerful **compound scaling method**  that uniformly scales up ConvNet in all three of the dimensions.
They use compound scaling coefficient  $$ \phi $$ to uniformly scale network width(w), depth(d), and resolution(r).

If we want to increase the baseline network by $$ 2^{\phi} $$ then we scale up $$ w $$, $$ d $$ and $$ r $$ in the following way.

$$d=\alpha^{\phi}$$

$$w=\beta^{\phi}$$

$$r=\gamma^{\phi}$$

$$\text{where   }   \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 $$ 

$$\text{and  }   \alpha \geq 1, \beta \geq 1, \gamma \geq 1 $$


### Explanation

If we increase ConvNet depth $$ d $$ by factor of $$ x $$  the network grows approximately $$ x $$ times. <br/>
If we increase ConvNet width $$ w $$ by factor of $$ x $$  the network grows approximately $$ x^2 $$ times. <br/>
If we increase ConvNet resolution $$ d $$ by factor of $$ x $$  the network grows approximately $$ x^2 $$ times. <br/>

$$ \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 $$ means that if we increase network by this combination of factors $$ \alpha, \beta, \gamma  $$ the network grows **twice** in terms of FLOPS. 

This leads to:
$$ \alpha^{\phi} \cdot (\beta^{\phi})^{2} \cdot (\gamma^{\phi})^{2} \approx 2^{\phi} $$

 
##  Baseline network search and finding $$ \alpha, \beta, \gamma $$

[Neural net architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search) is now becoming more popular for designing efficient mobile-size ConvNets. **Authors of this paper inspired by [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf) developed their baseline network EfficientNet-B0, which is clean and straightforward, making it easy to scale.**

**EfficientNet-B0** is baseline network, which means that:

$$\phi = 0 $$ and $$ d = {\alpha}^0 = 1,\  w = {\beta}^0 = 1,   \   r = {\gamma}^0 = 1$$ 

To obtain **EfficientNet-B1** where:

$$ \phi = 1 $$ and $$ d = {\alpha}^1, w = {\beta}^1, r = {\gamma}^1 $$ 

they now have to find the coefficients $$ \alpha, \beta,\text{and} \gamma$$ .

They do that with small grid search, where they check many variations of the coefficients and find the one which gives an EfficientNet-B1 with the best performance.

Scaling up further to obtain **EfficientNet-B2** to **EfficientNetB7** is a matter of tuning the compound scaling coefficient $$ \phi = 2 \ldots 7$$.


## Steps to come up with YourEfficientConvNet

Designing efficient ConvNets with compound scaling method is not limited to EfficientNet family. **In this paper, they give us enough tools to come up with our own efficient ConvNet.** 

Using manual tuning or neural architecture search, **you can come up with your own baseline network YourEfficientConveNet-0.** 

**And then you can search for coefficients $$ \alpha, \beta,\text{and} \gamma$$ that get the best accuracy for the specific problem you are trying to solve.**

