---
title: Compression Methods
category: posts
tags: research
---

There are plethora of model compression techniques used to compress very large trained neural networks for inference on edge devices, such as mobile phones and IoT devices. The purpose of these compression techniques is to fit the constraits of these edge devices. For example, an edge device may face memory constraints in storing the model weights, latency constraints for real-time applications, and energy constraints as these devices may be running on a limited power supply. 

To make inference of these models more efficient, there are several popular model compression techniques, namely quantization, pruning and knowledge distillation. In this post, we introduce some key papers surrounding these topics, and other methods of model compression.

## Quantization
Quantization of network weights involve approximating the float representation with a fewer bits. For example, by compressing the usual 32bit Floating Point (FP32) tensors into INT8, we can gain $\times 4$ reduction in model size and bandwidth requirements during inference. Binarization can be seen as an extreme version of quantization where the weights and activations are constrained to binary values.

### Quantization and training of neural networks for efficient integer-arithmetic-only inference (Jacob et al, 2018)
Quantization can be done post-training, where we train the original models using FP32 weights, then quantize the resulting 32-bit weights into smaller bit-width representations for inference. However, this leads to two common failure modes: 1) large differences (in the order of $$100$$) in ranges of weights between channels in a given CNN layer, as weights in the same layer are quantized to the same resolution, increasing relative errors for lower range activations; 2) outlier weight values making remaining weights more imprecise post-quantization. 

To alleviate these issues, Quantization Aware Training (QAT) involves altering the training scheme to simulate quantization of weights and biases during forward propagation pass. In Jacob et al., this is implemented on a CNN model by quantizing weights before convolution with input tensors, and the activations are quantized after convolutional/fully-connected layers or residual connections if any. 

To reduce error accumulation, biases of the layers are quantized with a bit-width larger than the weights (e.g. uint8 weights require 32-bits representations like int32), as each bias-vector value is added to many output activations (based on the convolved product of weights and input). This does not significantly impact the model size as biases account for only a tiny fraction of the model parameters. The QAT method allows trainig to be done using Integer Arithmetic instead of Floating Point Arithmetic, training up to $$\times 4$$ faster using only $$25\%$$ memory footprint.


### Quantized 8-bit BERT (Zafrir et al, 2019)
This paper applies QAT during the fine-tuning process of BERT by simulating 8bit quantized inference on all General Matrix Multiply (GMM) operations in the BERT fully connected and embedding layers (that comprise over $$99\%$$ of the model's weights), achieving $$\times 4$$ memory footprint while maintaining $$99\%$$ accuracy. The quantization scheme used is symmetric linear quantization, defined as follows:

$$\begin{align}
\text { Quantize }\left(x | S^{x}, M\right): &=\text { Clamp }\left(\left\lfloor x \times S^{x}\right\rceil,-M, M\right) \\
\text { Clamp }(x, a, b) &=\min (\max (x, a), b)
\end{align}$$

where $$S^x$$ is the quantization scaling factor for input $$x$$, $$\lfloor . \rceil$$ denotes rounding to the nearest integer,  and $$M$$ is the highest quantized value when quantized to $$b$$ bits with $$M = 2^{b-1} -1$$. The scaling factor for weights $$S^W$$ and values $$S^x$$ is calculated as follows, where EMA is the exponential moving average based on values encountered during training:

$$\begin{align}
S^{W}=\frac{M}{\max (|W|)} \\
S^{x}=\frac{M}{EMA(\max (|x|))} 
\end{align}$$

### Q-bert: Hessian based ultra low precision quantization of BERT (Shen et al, 2019)
[Shen et al](https://arxiv.org/abs/1909.05840) uses second order (i.e. Hessian) information of the weights in Transformer models to apply mixed-precision quantization, achieving compression ratios of $$\times 13$$ at up to $$2.3\%$$ accuracy degradation. To achieve higher compression ratios than that int8 quantization, mixed-precision quantization uses weights between 2, 4, or 8-bits in different layers. However, the search space for assignment of precision levels in layers is exponential in the number of layers, i.e. choosing between $$3$$ different bit-widths for a $$12$$ layer BERT model results in $$3^{12} \approx 5.3 \times 10^5$$. This is ameliorated by using a sensitivity measurement based on the mean and variance of top eigenvalues that minimizes quantization error. Concretely, layers with higher Hessian spectrum (i.e. larger top eigenvalues) are more sensitive to quantization, and thus are represented with highe precision. The Hessian matrix is computed using power iteration, and quantized according to the sum of the mean of the eigenvalues distribution and its standard deviation. 

The Q-BERT model also uses a different quantization scheme termed group-wise quantization, which partitions each matrix into different groups, each with its unique quantization range and lookup table. In layer-wise quantization, each of the $$4$$ weight matrix per attention head (key, query, value and output weights) are quantized with the same range. This causes an increase in error when using the same quantization range across weights with different ranges. To alleviate this, Q-BERT ungroups the weight matrices within each head, and istead groups sequential outputs of the same weight matrix together. This in effect assumes that the range of weights of neighbouring token in a partial sequence is lower than that of the $$4$$ weight matrices in each attention head.

The study also shows how the embedding layer is more sensitive to quantization than weights, in which positional embeddings are especially sensitive. This results highlights the importance of different quantization levels for varying modules.

## Pruning
Network pruning involves removing connections in the hidden layers of a network to reduce the number of parameters needed for inference. In the model-agnostic element-wise pruning approach, the weights of each neuron is ranked and pruned for those that produce the least effect on the activation values. Other methods of pruning involves reducing the number of components in a particular model, e.g. pruning the number of attention heads in Transformer models.

### Learning both Weights and Connections for Efficient Neural Networks (Han et al, 2015)
[Han et al](https://arxiv.org/abs/1506.02626) suggest an iterative method for pruning neural networks- after an initial training phase, weights below a threshold is dropped, turning the fully-connected layer into a sparse network. The sparse network is then re-trained to allow the remaining connections to learn to compensate for the previously dropped connections, with the dropping and re-training repeated iteratively until desired model size is obtained.

### Are Sixteen Heads Really Better than One? (Michel et al, 2019)
[Michel et al](https://arxiv.org/abs/1905.10650) show that in a transformer model, the standard size of $16$ attention heads can be pruned without significantly degrading performance accuracy, with some layers pruned up to $1$ head. The pruning step involves greedily and iteratively pruning attention heads with the least impact on model performance. Further study in the types of heads pruned show that encoder-decoder attention layers in machine translation are more sensitive to pruning than self-attention layers, providing evidence that different layers in the transformer model benefit more from having more heads.

### Rethinking the Value of Network Pruning (Liu et al, 2018)
[Liu et al](https://arxiv.org/abs/1810.05270) suggest that fine-tuning a pruned model is not superior to training the pruned model from scratch, implying that the pruned network architecture is far more important than the inherited weights.


## Knowledge Distillation
Knowledge distillation involves pre-training a large teacher network on an intended task, then training a smaller student network on the intermediate outputs of the teacher network \cite{hinton2015distilling}. When applied to transformer models, the student can potentially learn from the encoder outputs, logits in the final layer, or the attention maps.

### Distilling the Knowledge in a Neural Network (Hinton et al, 2015)
[Hinton et al](https://arxiv.org/abs/1503.02531) builds from [Bucilua et al](http://www.niculescu-mizil.org/papers/rtpp364-bucila.rev2.pdf) , whom initially introduce the concept of knowledge distillation by having a student network train from the logits of a teacher network, with an objective of minimizing the squared error of the logits produced by the two models. Hinton et al. then added a temperature parameter which is multiplied as a reciprocal to the output logits (making the input logits 'softer'). The student model is trained using the same temperature setting during distillation, but has its temperature reset to 1 (i.e. normal softmax) during inference. The training objective is a weighted average of the cross entropy loss between the soft targets of the two models, and the cross entropy loss with the correct labels. Performance was shown to be better when the weights are biased towards the first objective.

## Others
### Lite Transformer with Long-Short Range Attention (MIT HAN lab, 2020)
This paper propose a hybrid CNN-Attention module replacing the Self-Attention layers in Transformers, where the CNN and Attention layers capture local and global context respectively. This characterization reduces the computation required in the original Self-Attention layers, where O(d.N^2) can be replaced by Convolutions O(N.d^2.k), where N represents the length of input sequence, d represents the embedding dimension, asnd k represents the kernel size. Each input sequence is halfed along the embedding dimension d, and each half is fed into the Conv or Attention layer. Importantly, this paper defines constraints for NLP tasks on a mobile settings, derived from the computer vision community. These memory and computation constraints are set at 10M parameters and 500M Mult-Add (or 1G FLOPS) operations respectively. This is a good base for me to compare against, when evaluating mobile-constrained DNNs.

### Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers (Li et al, 2020)
For a given computational budget in inference, a larger model by width or depth (e.g. 3-layers vs 24-layers) that is heavily compressed outperforms a smaller model with lighter compression. This is because wider and deeper models converge in significantly fewer steps and are more robust to compression via pruning or quantization than smaller ones, making it more feasible to train the former when inference costs of computation far outweighs that of training costs. 

This paper show how training wider networks increases performance on machine translation tasks more effectively than training deeper networks, given the same increase in number of parameters. The work also shows how training efficiency plateaus off around a batch size of $$2048-16384$$. It is shown, however, that increasing batch size requires a corresponding increase in learning rate, whereas scaling model size does not require any subsequent hyperparameter adjustments.

## Conclusion
With a myriad of ways to compress models, the literature is diverse indeed. In fact, many of these methods are orthogonal to each other, where for example you can prune and quantize a student model trained with a knowledge distillation training set-up. This makes it really impactful when one develops a compression method, as it allows us to push the compression ratio limits of high-performant-high-size models today.

In my future research, I shall look into how to compress the popular BERT model for NLP tasks. In particular, what is the ideal model compression pipeline? Currently, popular methods include knowledge distillation (of say distilling BERT large into a 5X smaller model, e.g. DistillBERT), pruning (2-10X compression), and quantization (~4X compression). If we require chaining two to three methods to achieve a total compression ratio of 100X, in what order should the model compression algorithms be applied?