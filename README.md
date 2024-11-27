# DPZOPO
This is the implementation for the paper [Differentially Private Zeroth-Order Methods for Scalable Large Language Model Fine-tuning](https://arxiv.org/pdf/2402.07818). 
In this paper, we investigate the potential of DP zeroth-order methods for LLM pretraining, which avoids the scalability bottleneck of SGD by approximating the gradient with the more efficient zeroth-order gradient. We propose DP zeroth-order stagewise method (**DP-ZOSO**) and DP zeroth-order stagewise pruning method (**DP-ZOPO**) with several pruning strategies and undertake a comparative analysis of these strategies.

We conduct extensive empirical analysis on both encoder-only masked language model and decoder-only autoregressive language model, achieving impressive results in terms of scalability and utility regardless of the class of tasks.

<p>
  <img src="https://github.com/Liuxiaohao6/DPZOPO/main/assets/roberta-acc.png?raw=true" alt="Fig" width="100%"/>
  <em>
    
  </em>
</p>
