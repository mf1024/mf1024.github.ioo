---
layout: post
title: "Teaching GPT2 a sense of humor in PyTorch"
comments: true
---

Seeing the impressive results of the GPT2 generated texts, I wanted to play a little and decided to do a small project and finetune pretrained GPT2 model on jokes dataset and lets see if the GPT2 can come up with it's own jokes. 

Recently OpenAI team published a [paper](link to paper) on training a large [Transformer type](link to Attention is all you need) decoder model on language modelling task. They trained to predicte the next word in the 40GB(that is alot for a text dataset) of internet text dataset. They generated the dataset by following reddit links of upvoted posts, that way creating some filter for the quality of the posts.

The models they trained were much bigger than any previously trained models, and they used much more data than in any previously training sessions. 

-- some images of the decoder and transformer

-- The results were incredible. As you can see in the blog post. But remember that not all of the text generated was so cohorent and sense making. These texts are selecet. But the results are impressive nonetheless. 



-- I decided to play a little and do finetuning on jokes dataset. Now they have published all the models. 

-- huggingface have published their pretrained models.

-- lets first test if I have understood the outputs of the pretrained model correctly
  -- generate some tests / Matrix


-- I will use reddit jokes datase I found. 


-- Code.
-- Getting the model
-- Pretraining and hyperparams.
-- Present the results. 

