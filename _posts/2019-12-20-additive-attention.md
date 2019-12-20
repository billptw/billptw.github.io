---
layout: post
title: Additive Attention
---

Attention mechanism is a very popular technique used in neural models today, with many powerful variations. Today, we will look at additive attention ([Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473.pdf)), which was introduced as a solution to the fixed-sized hidden state problem of the seq2seq model.

## Why attention?

![Seq2Seq Model](/images/en2ch.png){:height="50%" width="50%"}{: .center-image }
*Source: ([Weng, 2018](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html))*

In the motivating example of translating a sentence from one language to another, the sequence-to-sequence model featuring an encoder and decoder is commonly used (refer to the [previous post]({{ site.baseurl }}{% link _posts/2019-12-18-Seq2Seq.md %}) for more in-depth explanation). In the seq2seq model, the decoder takes in an input sequence and generates a context vector of pre-defined length. For purposes of illustration, one can interpret the context vector as the 'meaning' of the input sentence that is captured by the neural network. For the above example of translating from English to Chinese, the sentence "She is eating a green apple" comprising 6 words (tokens) is represented as a context vector of 5 numbers. This context vector is then used at the input of another network, the decoder, to generate the desired translated sentence in Chinese.

What are some potential problems of this method? First, as the encoder is required to compress the representation of the entire input sequence into a fixed length context vector, resulting in some compressive loss. Additionally, as the encoder reads the input sequence one token at a time, by the time 456789. To calculate the error between the network output and ground truth label during training, the gradients of each node across all time-steps are multiplied (the process known as backpropagation through time). When multiplying across many time-steps, small or large values of the gradients are compounded (imagine what happens when you take 0.2 to the $$10^\{th\}$$ power), causing the gradients to converge at $$0$$ or saturate at $$\infty$$ and resulting in stalled training or NaNs (known as the vanishing and exploding gradient problem respectively).
