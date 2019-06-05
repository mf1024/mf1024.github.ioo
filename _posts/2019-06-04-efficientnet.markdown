---
layout: post
title: "EfficientNets explained"
---

Authors of the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) presents systematic method for finding efficient convolutional network architectures. Using this method they obtained family of networks called **EfficientNets** . 

**EfficientNets achieve dramatically better efficiency than previous ConvNets** including ResNets, DenseNets, Inception. 

**EfficientNet can achieve the same accuracy on ImageNet as other ConvNets by using up to 8.4x less parameters and up to 16x less FLOPS.** EfficientNet-B7 even achieves state-of-the-art 84.4% top-1/ 97.1% top-5 accuracy on ImageNet, while using 8.4x less parameters, and being 6.1x faster than the best existing ConvNet -  GPipe. 

![scaling](/img/efficientnet/scaling.png)

## The goal

**In this paper they investigate wether there is principled method to scale up the Convolutional networks that can achieve better accuracy and efficiency.**

There are many ways to scale up a ConvNet. To simplify the scaling problem first they fix some baseline ConvNet architecture (see the image above) and scale each network stage in three dimensions:
- width - number of channels
- depth - number of layers
- resolution - height times width of the tensor

**They define an optimization problem which they study: how to scale up the network in width, depth and resolution such that the network fits within given memory and FLOPS constraints and gives the best performance.**

## ConvNet scaling

In terms of efficiency, scaling up ConvNets is not well understood process because the optimal design search space is large and tuning process is usually slow, tedious and expensive. 

Most commonly networks are scaled in depth by adding more layers, but as they show in their study **scaling up in just one dimension is not optimal use of additional resources assigned to the ConvNet**. 

Experimentally they find that **it is critical to balance ConvNet scaling in all three of the dimensions** width, depth and height to achieve better performance and efficiency. 

##  Compound scaling method

They propose simple yet powerful **compound scaling method**  that uniformly scales up ConvNet in all three of the dimensions.
They use compound scaling coefficient  $$ \phi $$ to uniformly scale network width(w), depth(d), and resolution(r).

If we want to increase the baseline network by $$ 2^{\phi} $$ then we scale up $$ w $$, $$ d $$ and $$ r $$ in following way.

$$d=\alpha^{\phi}$$

$$w=\beta^{\phi}$$

$$r=\gamma^{\phi}$$

$$\text{where   }   \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 $$ 

$$\text{and  }   \alpha \geq 1, \beta \geq 1, \gamma \geq 1 $$


### Explanation

If we increase ConvNet depth $$ d $$ by factor of $$ x $$  the network will grow approximetly $$ x $$ times. <br/>
If we increase ConvNet width $$ w $$ by factor of $$ x $$  the network will grow approximetly $$ x^2 $$ times. <br/>
If we increase ConvNet resolution $$ d $$ by factor of $$ x $$  the network will grow approximetly $$ x^2 $$ times. <br/>

$$ \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 $$  simply means that if we increase network by this combination of factors $$ \alpha, \beta, \gamma  $$ the network will grow **twice** in terms of FLOPS. 

This leads to:
$$ \alpha^{\phi} \cdot \beta^{2 \cdot \phi } \cdot \gamma^{2 \cdot \phi} \approx 2^{\phi} $$

 
##  Baseline network search and finding $$ \alpha, \beta, \gamma $$

[Neural net architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search) is now becoming more popular for designing efficient mobile-size ConvNets. **Authors of this paper insipired by [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf) developed their own baseline network EfficientNet-B0 which is simple and clean, making it easy to scale.**

**EfficientNet-B0** is baseline network which means that 

$$\phi = 0 $$ and $$ d = {\alpha}^0 = 1,\  w = {\beta}^0 = 1,   \   r = {\gamma}^0 = 1$$ 

To obtain **EfficientNet-B1** where 

$$ \phi = 1 $$ and $$ d = {\alpha}^1, w = {\beta}^1, r = {\gamma}^1 $$ 

they now have to find the coefficients $$ \alpha, \beta,\text{and} \gamma$$ .

They do that with small grid search, where they check many variations of the coefficients and find the one which gives a EfficientNet-B1 with the best performance.

Scaling up further to obtain **EfficientNet-B2** to **EfficientNetB7** is matter of tuning the the compound scaling coefficient $$ \phi = 2 \ldots 7$$.


## Steps to come up with YourEfficientConvNet

Desiging efficient ConvNets with compund scaling method is not limited to EfficientNet family! **In this paper they give us enough tools to come up with our own efficient ConvNet.** 

Using manual tuning or neural architecture search **you can come up with your own baseline network YourEfficientConveNet-0.** (I recommend to use search.. it is crucial to have a good baseline model)

**And then you can search for ceofficients $$ \alpha, \beta,\text{and} \gamma$$ that get the best accuracy for the specific problem you are trying to solve!**
