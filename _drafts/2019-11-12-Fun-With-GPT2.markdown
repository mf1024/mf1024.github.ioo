---
layout: post
title: "Fine-tuning large Transformer models on single GPU in PyTorch - Teaching GPT-2 a sense of humor."
comments: true
---

## The model

Recently OpenAI team published an [article](https://openai.com/blog/better-language-models/) and a [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on training bigger and better language models to resarch language model abilities to generate coherent text and solve NLP tasks that the model were not explicitly trained for.

To do that they created [Transformer type](link to Attention is all you need) model which they call GPT-2 and trained it on a huge 40GB internet text dataset. 

The results they got at generating text are [very impressive](https://openai.com/blog/better-language-models/#sample1). The coherence of the text is human level as I cannot find any sign that these were written by a machine. 

## Fine tuning

I decided to play a little with it and try to teach the model to crack some jokes. 

I will use pretrained GPT-2 and [fine tune](http://wiki.fast.ai/index.php/Fine_tuning) on a jokes dataset. 

Thanks to generosity of teams who publish pretrained models, relatively cheap solutions for solving complex NLP tasks and high quality prototypes like this one  are possible. Training such models from scrach would costs tens of thousands of dollars, [in some cases even hundreds of thousands](https://syncedreview.com/2019/06/27/the-staggering-cost-of-training-sota-ai-models/). Fine tuning pretrained model on a new task might take few hours. 

GPT-2 comes in 4 different sizes, small, medium, large and [XL](https://openai.com/blog/gpt-2-1-5b-release/). With 124M, 355M, 774M and 1.5B parameters respectively. Medium I think is the largest that you can work with on single GPU so I will select medium. 



I will use PyTorch for this task. Huggingface have made many [pretrained Transformer](https://github.com/huggingface/transformers) models available for easy use in PyTorch so I will be using theirs.

Before fine-tuning the model on jokes, I will test it on generating a text to see if the weights were loaded and I have understood the outputs correctly.


## Testing the model

First install the huggingface pip package:

{% highlight shell %}
pip install transformers
{% endhighlight %}




## Jokes dataset

## Training in single GPU

## Results

-- some images of the decoder and transformer

-- lets first test if I have understood the outputs of the pretrained model correctly
