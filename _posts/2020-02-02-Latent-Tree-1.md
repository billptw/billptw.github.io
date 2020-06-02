---
title: Latent Tree Induction \#1
category: posts
tags: research
---

## Introduction
Now that it has been over a year of pursuing a PhD in AI, let's do a quick stock-take of what I have learnt thus far. Having come into the programme with virtually zero background knowledge nor experience in the AI field, I spent the first 4 months learning from courses in NTU, Coursera, Youtube, textbooks and various notes scattered online. This gave me a quick overview of the broad field of AI, while having enough knowledge to begin research. Thereafter, I spent the month of May in the Alibaba HQ in Hangzhou, to understand more about the company culture and met with my Alibaba mentors. This allowed me to align my PhD goals with the company requirements, where I would be focusing on running NLP applications on the edge.

The next 8 months or so was spent examining the latent tree induction line of research, which fascinates NLP researchers owing to its linguistics-backed inspiration. The importance of imbuing models with a hierarchical reasoning inductive bias is that language is naturally hierarchical, where words in a sentence relate to one another in a dependency or constituency parse. As improving model inductive bias is central to improving the accuracy in models without incurring additional parameters or cost, I studied this line of work for the past six months, and worked on two papers in this topic. Here, I am sharing my notes here today, and will spread it over two posts to minimize verbosity. Would love your comments and feedback!

## Trellis for Sequence Modelling (ICLR 2019)
Temporal Convolutional Networks (TCNs) merges recurrent models with convolution, using 1-D convolution where hidden layers and input layers have equal length (mimicking hidden-hidden layuers in an un-rolled RNN), and convolutions occur without memory leak from the future. In a Trellis Network (T-Net), it takes the TCN one step further by tying weights not just along the layer ('time-steps' in the un-rolled RNN analogy), but across layers to form a trellis pattern of weights. Further, the inputs are injected into all network layers. While seemingly bizarre, this novel architecture actually performs well on baseline tasks like word-level PTB (54.19 ppl vs LSTM 73.4),  and word-level WikiText-103 (29.19 ppl vs LSTM @ 48.7). The modelling ability comes from the two unique features: weight tying in a trellis-formation is a form of regularization which improves stability during training, while network-wide input injection mixes deep features with the original sequence. 

## An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling(Bai et al, 2018)
Temporal Convolutional Networks (TCN) are useful in sequence modelling tasks, doing well on benchmark recurrent network tasks such as the adding problem, copy memory tasks, etc., despite not relying on recurrent architectures. This convolutional architecture, basically a 1D fully-convolutional network with casual convolutions (convolutions only on previous timesteps), has the following properties: 1) the architecture can take a sequence of any length and map it to an output sequence of the same length, just as with an RNN; 2) the convolutions in the architecture are causal, meaning that there is no information “leakage” from future to past.

## The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives (2019)
Wanting to understand the objective functions of different NLP tasks, this paper analysis the representations of individual tokens and layers and their interactions, and how that changes as the Transformers model undergoes training. One particularly interesting insight into the LM objective function is in how as we traverse the layers from bottom to top, information about the past gets lost and predictions of the future is formed. This shows how the layers have some latency effect in transmitting information of the present token, and a heirarchical storage in memory (tokens from many time-steps before) layer-wise.

## Tree LSTMs with Convolution Units to Predict Stance and RumorVeracity in Social Media Conversations (ACL, 2019)
This paper models social media conversational threads as trees, then uses the TreeLSTM model in the rumor classification in social media posts, used to detect 'fake news'. This is in contrast with previous models using a Branch LSTM design, where encodings of source-tweets form the root with replies following from branches; the Tree LSTM allows further branching where replies to replies of source tweets can be added hierarchically, instead of just sequentially as in Branch LSTM (one parent many children). Another advantage of the TreeLSTM structure is in the ability to weigh between branches in the tree to attibute more value to more informative branches.

## Learning to Compose Words into Sentences with Reinforcement Learning (2016)
This model uses RL to learn the tree structure in the sentence representation model of the SPINN (Stack-augmented Parser-Interpreter Neural Network) architecture. This parser is stack-augmented from its reliance on a stack for the Shift-Reduce algorithm employed during parsing; the Shift operation serves to push each token into the stack as an input sentence is parsed (handled by a tranversing pointer), while the Reduce operation pops two elements from said stack to compose into one, which is then pushed back. Corresponding to the parse tree, the Shift phase adds a new leaf node into the tree while the Reduce action merges a node pair into a constituent. The RL algorithm is then used to learn the ideal Shift-Reduce policy, where Shift and Reduce steps are actions for the agent to take. Concretely, RL-SPINN uses the REINFORCE algorithm with a reward function being the probability of predicting the correct label using a sentence representation composed in the order given by the sequence of actions sampled from the policy network.

## Unsupervised Recurrent Neural Networks Grammars (2019)
Lack of independence assumption allows models to perform well on language modeling, but not on grammar inductive takes like unsupervised constituency parsing.

## Compound Probabilist Context-Free Grammars for Grammar Induction (2019)
This model learns grammar in a compound context-free rule probabilistic way, over the traditional method of a single stochastic grammar induced. The model, while performing badly on the language modeling task (>190 ppl on PTB),  has better trees when evaluated on the unsupervised parsing task (F1 score 60.1 over ON-LSTM at 49.4).

## From Softmax to Sparsemax (2016)
The Sparsemax is a alternate Softmax formulation which is especially useful in making Transformers models sparser. This is done by obtaining the Euclidean projection of an input vector into a probability simplex (n-dimensional triangle), making it more likely to map low values to zero.

## Adaptively Sparse Transformers (2019)
This model introduces sparsity to attention heads of the Transformers model by using an alternate Softmax function, termed the alpha-entmax. This alternate Softmax assigns a weight of 0 for low-scoring words, improving interpretibility and attention head diversity in machine translation. This sparsely introduces diversity in the specialisation of attention distributions between layers, allowing model interpretation at zero performance loss. This is in contrast with previous models using the sparsemax in two ways: 1) The current proposed model is adaptively sparse, where the sparsity is a learnable parameter; 2) The heads can attend to a non-contiguous span of tokens, VS only on contiguous spans.

## On Controllable Sparse Alternatives to Softmax (2018)
This paper combines previously proposed Softmax functions like softmax, sum-normalization, spherical softmax, and sparsemax, into a unified framework encompassing all, termed the sparsegen-lin and sparsehourglass.

## Levenshtein Transformer (2019)
This paper proposes a Transformers model with insertion and deletion mechanisms, allowing the decoder to edit the hidden representation during decoding. This is inspired by the human-like abilities to revise generated text during creation, which previous models lack, wherein generation and refinement are treated as two different tasks. The model is trained by imitation learning, producing the two (complementary yet adversarial) policies of inserting and deleting text that are executed in succession.

## Mogrifier LSTM (2019)
The latest SOTA architecture on language modeling (with a funky name to boot). In essence, it introduces a mutual gating mechanism between current and previous input, making the transition function context-dependent. The mutuality in gating is essential, where previous literature (Multiplicative LSTM) only provides one-way gating. An interpretation in the improved representational abilities provided by the authors is context-laden representations of input token (conditioning input embeddings on recurrent state).
