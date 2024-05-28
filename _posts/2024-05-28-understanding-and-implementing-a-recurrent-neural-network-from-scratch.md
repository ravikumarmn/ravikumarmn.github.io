---
title: 'Understanding and Implementing A Recurrent Neural Network'
author: Ravikumar
date: 2024-05-27T19:00:49.679Z
categories:
  - Natural Language Processing
  - Recurrent Neural Network
tags:
  - Recurrent Neural Network
  - Embeddings
image:
  path: images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/rnn_unrolled.png
  alt: RNN Unrolled.
permalink: /blogs/:title
comments: true
---

This blog helps you to understand the Recurrent Neural Network and implement Recurrent Neural Network using Pytorch.

**Recurrent Neural Network's behaviour is to remember the information for periods of time.**

How it Remembers the information?
It creates the networks with loops in them, Which allows it to carry the information.

![rnn_rolled.png](images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/rnn_rolled.png)


In the above diagram, A piece of neural network, **R**, Takes a input Xt and outputs a value Ht. Loops allows the information to be passed from one step of network to another step of network.

You can think of A recurrent neural network architecture, comprise of multiple copies of same network connected to each other and passing the information from one to another.

![rnn_unrolled.png](images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/rnn_unrolled.png)

The basic equations that defines RNN is shown below
![rnn_formula.png](images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/rnn_formula.png)

Where, xt is input at time t,ht is hidden state at time t,h(t-1) is hidden state of previous layer at time t-1 or the initial hidden state.

Here, If we look into our architecture, It takes input at time t and hidden state of previous layer at time t-1, does the transformation/calculation and gives us output to use it for next network.

![single_rnn.png](images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/single_rnn.png)


Same repeat for every time steps.

For those of you who like looking at code here is some code.
```python

rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
```
        
Now let's build Recurrent Neural Network using PyTorch framework.

```python
class MyRNN(nn.Module):
    '''Building RNN using PyTroch'''
    def __init__(self, input_size, hidden_size):
        '''
        Args : 
            input_size :Size of embeddings. (batch,seq,embeddings)
            hidden_size : Hidden size, to transform the embedding to this shape

        Returns :
            output : output features (h_t) from the last layer of the RNN
            hn  : final hidden state
        '''        
        super(MyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
    def calulate_ht(self,x,prev):
      '''calculate ht using formula given above'''
        wih = self.ih(x)                    # eq1 = xt*Wih
        whh = self.hh(prev)                 # eq2 = h(t-1) * Whh
        combined = torch.add(wih, whh)      # eq3 = eq1 + eq2
        hidden_state = torch.tanh(combined) # tanh(eq3)
        return hidden_state 

    def forward(self, x):
        batch_sz,seq_sz,_ = x.size()
        prev = torch.zeros(batch_sz, self.hidden_size)
        hidden_seq = list()
        for i in range(seq_sz):
            xt = x[:,i,:]
            prev = self.calulate_ht(xt,prev)
            hidden_seq.append(prev)
        hn = hidden_seq[-1].view(1,batch_sz,-1)
        output = torch.stack(hidden_seq,dim = 1).view(batch_sz,seq_sz,-1)
        return output,hn
```
I have implemented RNN model and also I have verified the output comparing with PyTorch nn.RNN module. 

Below are the results of output features (h_t) from the last layer of the RNN.
I am comparing my RNN built from scratch results with Pytorch nn.RNN module results. 

![rnn_out_compare.png](images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/rnn_out_compare.png)

If we see the results. both output features of last layer are same.

Now let's see the results and compare for **final hidden state** for each element in the batch.

![rnn_hn_results.png](images/Machine Learning/Natural_Language_Processing/RNN_SCRATCH/rnn_hn_results.png)

Great!, All value are same. We did it. Below is code. You can checkit out.

This way the RNN works.

RNN helps wherever we need context from the previous input.

Source code : [GitHub Code](https://github.com/ravikumarmn/Learning-NLP-with-PyTorch/blob/main/Chapter%20-%203%20:%20Build%20LSTM/rnn_scratch.py)

If you want to know, how you can use RNN/LSTM's to train your model.
Here is the project of binary classification using LSTM : [Sentiment Classificaiton and Word Embedding](https://ravikumarmn.github.io/blogs/sentiment-classification-and-word-embedding)
