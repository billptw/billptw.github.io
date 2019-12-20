---
layout: post
title: Additive Attention
---

Attention mechanism is a very popular technique used in neural models today, with many powerful variations. Today, we will look at additive attention ([Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473.pdf)), which was introduced as a solution to the fixed-sized hidden state problem of the seq2seq model.

## Why attention?

![Seq2Seq Model](/images/en2ch.png){:height="80%" width="80%"}{: .center-image }
*Source: ([Weng, 2018](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html))*

In the motivating example of translating a sentence from one language to another, the sequence-to-sequence model featuring an encoder and decoder is commonly used (refer to the [previous post]({{ site.baseurl }}{% link _posts/2019-12-18-Seq2Seq.md %}) for more in-depth explanation). In the seq2seq model, the decoder takes in an input sequence and generates a context vector of pre-defined length. For purposes of illustration, one can interpret the context vector as the 'meaning' of the input sentence that is captured by the neural network. For the above example of translating from English to Chinese, the sentence "She is eating a green apple" comprising 6 words (tokens) is represented as a context vector of 5 numbers. This context vector is then used at the input of another network, the decoder, to generate the desired translated sentence in Chinese.

What are some potential problems of this method? The main failure mode for seq2seq models arises from when the input sequence is long. To quote the previous example, by the time last token in the sentence, "apple", is fed into the encoder, the context vector (that has thus far encoded the first 5 words) would lose the representation for the first token "She". When this contect vector is subsequently fed into the decoder, the decoder essentially be forced to guess the first word in the source language, resulting in poor translations. 

This arises from a known problem in training RNN-based models (the vanishing/exploding gradient problem), explained in further detail: To calculate the error between the network output and ground truth label during training, the gradients of each node across all time-steps are multiplied (a process known as backpropagation through time). When multiplying across many time-steps, small or large values of the gradients are compounded (imagine what happens when you take 0.2 to the $$10^{th}$$ power), causing the gradients to converge at $$0$$ or saturate at $$\infty$$ and resulting in stalled training or NaNs (known as the vanishing and exploding gradient problem respectively). This failure mode in RNN-based encoder makes it hard for the model to properly encode for long-term dependencies between tokens many time-steps away (e.g. the first and last words in an input sentence), resulting in a context vector that represents the input sentence poorly.


