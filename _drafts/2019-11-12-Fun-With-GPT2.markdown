---
layout: post
title: "Fine-tuning large Transformer models on single GPU in PyTorch - Teaching GPT-2 a sense of humor."
comments: true
---

## The model

Recently OpenAI team published an [article](https://openai.com/blog/better-language-models/) and a [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on training bigger and better language models to resarch language model abilities to generate coherent text and solve NLP tasks that the model was not explicitly trained for.

To do that they created a [Transformer type](link to Attention is all you need) model which they call GPT-2 and trained it on a huge 40GB internet text dataset. 

The results they got at generating text are [very impressive](https://openai.com/blog/better-language-models/#sample1). The coherence of the text is human level as I cannot find any sign that these were written by a machine. 

## Fine tuning

I decided to play a little with the GPT-2 and try to teach the model to crack some jokes. 

I will use pretrained GPT-2 and [fine tune](http://wiki.fast.ai/index.php/Fine_tuning) it on a jokes dataset. 

Thanks to generosity of teams who publish pretrained models, relatively cheap solutions for solving complex NLP tasks are possible, as well as fast prototyping experiments like this one. Training such models from scrach would costs tens of thousands of dollars, [in some cases even hundreds of thousands](https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/). Fine tuning pretrained model on a new task might take few hours. 

GPT-2 comes in 4 different sizes, small, medium, large and [XL](https://openai.com/blog/gpt-2-1-5b-release/), with 124M, 355M, 774M and 1.5B parameters respectively. Medium I think is the largest that you can work with on a single GPU with reasonable sequnece sizes, so I will select the medium. 


I will use PyTorch for this task. Huggingface have made many [pretrained Transformer](https://github.com/huggingface/transformers) models available for easy use in PyTorch.


## Testing the pretrained model by generating text

Before fine-tuning the model on jokes, I will test it on generating a text to see if the weights were loaded with no issues.

In the following gist I will try to generate some text by using pretrained medium size GPT2 using the following beginnings of the text:

> ***\' The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth... \'***

> ***\' Artificial general intelligence is... \'***

> ***\' The Godfather: "I'm going to make him an offer he can't refuse."... \'***

{% gist 430d7fd6ff527350d3e4b5bda0d8614e %}



Judging by the generated conspiricy theories about technology, threarening predictions about AI industry and The Godfather dialoge with himself I would say that the text generation is working. 


## Jokes dataset



## Training on a single GPU

## Results

-- some images of the decoder and transformer

-- lets first test if I have understood the outputs of the pretrained model correctly
