# DPZOPO
This is the implementation for the paper [Differentially Private Zeroth-Order Methods for Scalable Large Language Model Fine-tuning](https://arxiv.org/pdf/2402.07818), which is accepted by **NDSS 2025**.
In this paper, we investigate the potential of DP zeroth-order methods for LLM pretraining, which avoids the scalability bottleneck of SGD by approximating the gradient with the more efficient zeroth-order gradient. We propose DP zeroth-order stagewise method (**DP-ZOSO**) and DP zeroth-order stagewise pruning method (**DP-ZOPO**) with several pruning strategies and undertake a comparative analysis of these strategies.

We conduct extensive empirical analysis on both encoder-only masked language model and decoder-only autoregressive language model, achieving impressive results in terms of scalability and utility regardless of the class of tasks.

<p>
  <img src="https://github.com/Liuxiaohao6/DPZOPO/blob/main/assets/roberta-acc.png?raw=true" alt="Fig" width="100%"/>
  <em>
    Accuracy on RoBERTAa-large between zero-shot, DP full-parameter fine-tuning (FT), DP prefix-tuning (FT-prefix), DPZero, DP-ZOSO and DP-ZOPO, .
  </em>
</p>

<p>
  <div align=center><img src="https://github.com/Liuxiaohao6/DPZOPO/blob/main/assets/memory.png?raw=true" alt="Fig" width="70%"/></div>
  <em>
    GPU memory usage and running time comparison between zero-shot, in-context learning (ICL), DP full-parameter fine-tuning (FT), DP prefix-tuning (FT-prefix), DP-ZOSO, DP-ZOPO.
  </em>
</p>


## Citation

```bibtex
@article{liu2024differentially,
  title={Differentially Private Zeroth-Order Methods for Scalable Large Language Model Finetuning},
  author={Liu, Zhihao and Lou, Jian and Bao, Wenjie and Li, Bo and Qin, Zhan and Ren, Kui},
  journal={arXiv preprint arXiv:2402.07818},
  year={2024}
}
```
