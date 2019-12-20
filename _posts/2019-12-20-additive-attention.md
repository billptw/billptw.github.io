---
layout: post
title: Additive Attention
---

Attention mechanism is a very popular technique used in neural models today, with many powerful variations. Today, we will look at additive attention ([Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473.pdf)), which was introduced as a solution to the fixed-sized hidden state problem of the seq2seq model.

## Why attention?

![Seq2Seq Model](/images/en2ch.png){:height="50%" width="50%"}{: .center-image }
*Source: ([Weng, 2018](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html))*{: style="text-align:center"}

In the motivating example of translating a sentence from one language to another, the sequence-to-sequence model featuring an encoder and decoder is commonly used (refer to the [previous post]({{ site.baseurl }}{% link _posts/2019-12-18-Seq2Seq.md %}) for more in-depth explanation). In this model, 
