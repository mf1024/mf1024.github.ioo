---
layout: post
title: "Teaching GPT-2 a sense of humor - Fine-tuning large Transformer models on a single GPU in PyTorch."
comments: true
---

![laughing_kid](/img/gpt2_finetuning/laugh.jpg)
<sup>Photo by [Ben White](https://unsplash.com/@benwhitephotography?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/)</sup>


In this post, I demonstrate how you can use pre-trained GPT-2 to generate text and then fine-tune it on a specific language modeling task using a single GPU. In this case, I try to teach the model to be funny by fine-tuning it on a jokes dataset. 


# The GPT-2 

Recently OpenAI team published an article [Better Language Models](https://openai.com/blog/better-language-models/), and a technical paper [Language Models Are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) about training bigger and better language models. They research language model abilities to generate coherent text and solve NLP tasks in a zero-shot setting, which means using the model to solve tasks that it was not explicitly trained for.

![open_ai](/img/gpt2_finetuning/open_ai.png)
<sup>Image Credit: OpenAI</sup>

They created a [transformer-based](https://arxiv.org/abs/1706.03762) language model that they called GPT-2 and trained it on a huge 40GB internet text dataset. They trained the model on a language modeling task, which is predicting probabilities of the next word in a word sequence. Training NLP models for language modeling and then [fine-tuning](http://wiki.fast.ai/index.php/Fine_tuning) for a specific task is one of the most common paths for training NLP models. Pre-training a model for language modeling is convenient because it does not require labeled data to learn the structure of language - it only requires plain text, which is openly available in vast amounts. Most publicly available pre-trained NLP models are trained for language modeling.

The results they got at generating text after the training are [very impressive](https://openai.com/blog/better-language-models/#sample1); the fragments feel very human and coherent that it's almost creepy. Also, the model achieved state-of-the-art scores in zero-shot settings on a variety of language modeling tasks, including summarization, reading comprehension, and translation. 

# Fine-tuning experiment plan

So I decided to experiment a little with the GPT-2. I thought it would be fun to teach the model to crack some jokes. To do that, I need a jokes dataset and a pre-trained GPT-2 model for fine-tuning.

![planning](/img/gpt2_finetuning/planning.jpg)
<sup>Photo by [Helloquence](https://unsplash.com/@helloquence?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/)</sup>


Thanks to the generosity of the AI community and some specific teams who publish pre-trained neural network models, relatively cheap solutions for solving challenging tasks like this one are possible. Training such large neural-network models from scratch would costs tens of thousands of dollars, [in some cases, even hundreds of thousands](https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/). Fine-tuning a pre-trained model on a new task might take a few hours on a single GPU. And I'll do just that.

Huggingface has made many [pre-trained Transformer](https://github.com/huggingface/transformers) models available for easy use in PyTorch. I'll use their pre-trained GPT-2 and fine-tune it on this [Short Jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes) published on Kaggle.

GPT-2 comes in 4 different sizes - small, medium, large, and [XL](https://openai.com/blog/gpt-2-1-5b-release/), with 124M, 355M, 774M, and 1.5B parameters, respectively. 
I found that a medium-size GPT-2 model is the largest of the models that I could fine-tune with reasonable input sequence length on a single GPU.

![gpt2-sizes](/img/gpt2_finetuning/gpt2-sizes.png)
<sup>Image Credit: Image by [Jay Alammar](https://jalammar.github.io) from post [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)</sup>


# Testing the pre-trained model by generating text

Before fine-tuning the model on jokes, I'll test it on generating a text.

In the following gist, I demonstrate how to generate a text by using pre-trained medium-size GPT-2 from huggingface. I'll feed the model the following text fragments to start with and let it generate the rest:

> ***\' The Matrix is everywhere. It is all around us. Even now, in this very room. You can see it when you look out your window or when you turn on your television. You can feel it when you go to work... when you go to church... when you pay your taxes. It is the world that has been pulled over your eyes to blind you from the truth... \'***

> ***\' Artificial general intelligence is... \'***

> ***\' The Godfather: "I'm going to make him an offer he can't refuse."... \'***

{% gist 430d7fd6ff527350d3e4b5bda0d8614e %}


**Judging by the generated conspiracy theories about technology, threatening predictions about the AI industry, and The Godfather dialogue with himself, I would say that the text generation is working.** 

# Fine-tuning the model on a single GPU

Large Transformer models are usually trained in multi-GPU(or TPU) settings because training on reasonable batch size and sequence length on a  large model requires lots of tensor/graphical processing unit memory. My machine is equipped with a single GeForce 1080 Ti, which has 11 GB of memory. By empirical tests on the medium-size GPT-2 model, I found that the maximum total sequence element count in a batch for my GPU to process is approximately 550, which is not a lot and might not be sufficient for successful fine-tuning.

But there are some things we can take into account to improve the situation.

The first thing to notice is that the batch size in a forward-backward pass of a transformer-based model does not play a role because [Layer Normalization](https://arxiv.org/abs/1607.06450) is used instead of Batch Normalization. In Layer Normalization, each feature is normalized across the [feature dimension](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/), and the batch dimension is not involved.

Second, we can accumulate gradients over multiple forward-backward passes, and only then do the model weight update. This way, we don't have to store the computational graph of a whole batch in the memory, but we can process sequence by sequence and achieve the same result as if the whole batch would have been processed in a single forward-backward pass.

Taking it all into account, I'll process one sequence at a time with a maximum length of 550 and do model weight update every *BATCH_SIZE* processed sequences. 

The length of jokes varies a lot in the dataset - there are many short sequences. To make the total sequence element count in one optimization step more consistent, I'll try to fit in as many jokes as possible in each 550 element sequence.

{% gist 3df214d2f17f3dcc56450ddf0d5a4cd7 %}


# Results and conclusions

![laughing_toy](/img/gpt2_finetuning/laughing_toy.jpg)
<sup>Photo by [Marcela Rogante](https://unsplash.com/@marchuri?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/)</sup>

It is a hard problem to teach AI to generate a text that'll seem funny to a human, and I think that it is much harder than to generate a coherent text. Even for a human, it is not easy to do - it takes a special kind of creativity, understanding of the context, and even understanding of human psychology. Feeding many jokes to a language model might not be sufficient for the model actually to learn what makes something funny. It might require more sophisticated techniques and a lot more data to train human-level joking models. 

**Nevertheless, it is hilarious to see this language model trying. Once in awhile, the model manages to generate a funny human-level joke.** 

*\*When I started the experiment, I did not notice that a significant portion of the jokes in the dataset are racist and rude, which means you can expect the same in the generated joke list from the model. I apologize for that and be prepared.* 

**Here is [the full generated jokes list](https://github.com/mf1024/transformers/blob/master/generated_2_jokes.txt).**

If you see something good and funny in the generated jokes list, post it in the comments. :) I didn't read through all of them myself.
