---
layout: post
title: "Fine-tuning large Transformer models on single GPU in PyTorch - Teaching GPT-2 a sense of humor."
comments: true
---

## The model

Recently OpenAI team published an article [Better Language Models](https://openai.com/blog/better-language-models/) and a paper [Language Models Are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) about training bigger and better language models to resarch language model abilities to generate coherent text and solve NLP tasks in zero-shot setting which means using the model to solve tasks that the model was not explicitly trained for.

They created a [transformer-based](https://arxiv.org/abs/1706.03762) language model which they called GPT-2 and trained it on a huge 40GB internet text dataset. 

The results they got at generating text are [very impressive](https://openai.com/blog/better-language-models/#sample1) and feels close to human quality. Also the model achieved state-of-the art scores on a variety of language modeling tasks in zero-shot setting. 

## The experiment plan

I decided to experiment a little with the GPT-2. I thought it would be fun to try to teach the model to crack some jokes. 

I will use pretrained GPT-2 and [fine tune](http://wiki.fast.ai/index.php/Fine_tuning) it on a [jokes dataset](https://github.com/taivop/joke-dataset)

Thanks to generosity of teams who publish pretrained models, relatively cheap solutions for solving complex NLP tasks as well as fast prototyping experiments like this one are possible. Training such models from scrach would costs tens of thousands of dollars, [in some cases even hundreds of thousands](https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/). Fine tuning pretrained model on a new task might take a few hours on a single GPU. 

GPT-2 comes in 4 different sizes - small, medium, large and [XL](https://openai.com/blog/gpt-2-1-5b-release/), with 124M, 355M, 774M and 1.5B parameters respectively. 

![gpt2-sizes](/img/gpt2_finetuning/gpt2-sizes.png)
<sup>Image source: an exceptional post [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), which I highly recommend to read.</sup>

I found that medium sized GPT2 is the largest of the models that I could fine-tune with reasonable sequence lenght and results on a single GPU.

I will use PyTorch for this task. Huggingface have made many [pretrained Transformer](https://github.com/huggingface/transformers) models available for easy use in PyTorch.


## Testing the pretrained model by generating text

Before fine-tuning the model on jokes, I will test it on generating a text to see if the weights were loaded with no issues.

In the following gist I will try to generate some text by using pretrained medium size GPT2 using the following beginnings of the text:

> ***\' The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth... \'***

> ***\' Artificial general intelligence is... \'***

> ***\' The Godfather: "I'm going to make him an offer he can't refuse."... \'***

{% gist 430d7fd6ff527350d3e4b5bda0d8614e %}

Judging by the generated conspiricy theories about technology, threarening predictions about AI industry and The Godfather dialoge with himself I would say that the text generation is working. 

## Fine-tuning the model

For fine-tuning I will use the Redit jokes from [this](https://github.com/taivop/joke-dataset) dataset. 



## Results

-- some images of the decoder and transformer

-- lets first test if I have understood the outputs of the pretrained model correctly
