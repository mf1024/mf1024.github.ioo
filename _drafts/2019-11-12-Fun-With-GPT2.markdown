---
layout: post
title: "Fine-tuning large Transformer models on a single GPU in PyTorch - Teaching GPT-2 a sense of humor."
comments: true
---

#TODO need first paragraph that better describes what will I try to achieve

In this post I will demonstrate how you can use pretrained GPT-2 to generate text and then fine-tune it on a specific language modeling task 

# The GPT-2 

Recently OpenAI team published an article [Better Language Models](https://openai.com/blog/better-language-models/) and a paper [Language Models Are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) about training bigger and better language models to resarch language model abilities to generate coherent text and solve NLP tasks in zero-shot setting which means testing the abilities of the model to solve tasks that it was not explicitly trained for.

They created a [transformer-based](https://arxiv.org/abs/1706.03762) language model which they called GPT-2 and trained it on a huge 40GB internet text dataset. GPT-2 was trained on language modeling task which is predicting probabilies of the next word in a word sequence. Training NLP models for language modeling and then fine-tuning for a sepcific task is one of the most common NLP model training paths because language modelling does not require labeled data to learn the structure of language, it only requires palin text which is openly available in vast amounts. All published and publicly available pretrained NLP models are trained on language modeling. 

The result they got at generating text after the training is [very impressive](https://openai.com/blog/better-language-models/#sample1), it feels very human and coherent it's almost creepy. Also the model achieved state-of-the art scores in zero-shot setting on a variety of language modeling tasks including summarization, reading comprehension and translation. 

# Fine-tuning experiment plan

So I decided to experiment a little with the GPT-2. I thought it would be fun to try to teach the model to crack some jokes. To do that I will need a jokes dataset and a pretrained GPT-2 model for [fine-tuning](http://wiki.fast.ai/index.php/Fine_tuning).  

Thanks to generosity of teams who publish pretrained neural-network models, relatively cheap solutions for solving challenging tasks like this one are possible. Training such lasrge neural-network models from scrach would costs tens of thousands of dollars, [in some cases even hundreds of thousands](https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/). Fine-tuning pretrained model on a new task might take a few hours on a single GPU. And I will do just that.

Huggingface have made many [pretrained Transformer](https://github.com/huggingface/transformers) models available for easy use in PyTorch.

I will use pretrained GPT-2 from huggingface and fine-tune it using Reddit jokes from this [jokes dataset](https://github.com/taivop/joke-dataset)

GPT-2 comes in 4 different sizes - small, medium, large and [XL](https://openai.com/blog/gpt-2-1-5b-release/), with 124M, 355M, 774M and 1.5B parameters respectively. 
I found that medium-sized GPT-2 model is the largest of the models that I could fine-tune with reasonable input sequence lenght on a single GPU.

![gpt2-sizes](/img/gpt2_finetuning/gpt2-sizes.png)
<sup>Image source: [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), which is an exceptional post and I highly recommend to read it.</sup>


# Testing the pretrained model by generating text

Before fine-tuning the model on jokes, I will test it on generating a text.

In the following gist I demonstrate how to generate a text by using pretrained medium size GPT2 from huggingface. I will give the model the following text fragments to start with and let it generate the rest:

> ***\' The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth... \'***
![morpheus](/img/gpt2_finetuning/morpheus.jpg)

> ***\' Artificial general intelligence is... \'***
![asimov](/img/gpt2_finetuning/asimov.jpg)

> ***\' The Godfather: "I'm going to make him an offer he can't refuse."... \'***
![the_godfather](/img/gpt2_finetuning/the_godfather.jpg)

{% gist 430d7fd6ff527350d3e4b5bda0d8614e %}


**Judging by the generated conspiricy theories about technology, threarening predictions about AI industry and The Godfather dialoge with himself I would say that the text generation is working.** 

# Fine-tuning the model on a single GPU

Large Transformer models are usually trained in multi-GPU(or TPU) settings because training on reasonable batch size and sequence length requires lots of tensor/graphical processing unit memory. My machine is equipped with a single GeForce 1080 Ti, which has 11 GB of memory. By empirical tests on the medium sized GPT-2 model I found that the maximum total sequence element count in a batch for my GPU to process is approximately 550, which is not a lot and might not be sufficient for successful fine-tuning.

But there are some things we can take into account and improve the situation.

The first thing to notice is that the batch size in forward-backward pass of a transformer-based models does not play a role because [Layer Normalization](https://arxiv.org/abs/1607.06450) is used instead of Batch Normalization. In Layer Normalization, each feature is normalized across the [feature dimension](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/).

Second, we can accumulate gradients over multiple forward-backward passes, and only then do the model weight update. This way, we don't have to store computational graph of a whole batch in the memory, but we can process sequence by sequence and achieve the same result as if whole batch would have been processed in a single forward-backward pass.

Taking it all into account I will process one sequence at a time with a maximum length of 550. 

The length of joke sequences varies a lot in the dataset - there are many short sequences. To make the total sequence element count in one optimization step more consistent, I will try to fit in as many jokes as possible in the 550 element sequence.

{% gist 3df214d2f17f3dcc56450ddf0d5a4cd7 %}


# Results and conclusions

It is a tricky problem to teach AI to generate a text that will seem funny to a human and I think that it is much harder paroblem than to generate a coherent text. Feeding many jokes to a language model and fine-tuning it might not be sufficient for the model to actually learn what makes something funny. It might require more sophisticated techiques and a lot more data to train human-level joking models. 

**Nevertheless it is very funny to see this model trying. Once in a while the model manages to generate a funny human level joke.** 

***\*What I did not notice when I started the experiment is that a big portion of the Reddit jokes are racist and even evil, which means you can expect the same in the generated joke list from the model and I apologize for that. If you think something is so bad it should not even be in the list, write me a message and I will remove it.*** 

Here are some funny examples:

Here is [full generated joke list]()


