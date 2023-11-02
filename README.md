# Low-Rank LLama2

In the ever-evolving landscape of artificial intelligence (AI), one undeniable trend has emerged in recent years: the relentless growth in the size and complexity of machine learning models. More specifically, large language models (LLMs) that mainly rely on transformers as building blocks, are reaching a substantial number of parameters and require a significant amount of compute that is expected to increase with larger and larger models being released.

In this blog post and supporting code, we explore low-rankness as a pruning technique of the LLama2-7B base model. We show that, by splitting almost all the linear layer weights into low-rank pairs without fine-tuning and leveraging LoRA for custom training, we can achieve the following without implementing custom kernels:

- ~50% reduction in the number of parameters.
- Up to ~50% faster training vs. bitsandbytes’s 8-bit quantization.
- Up to ~1.25x inference speed-up.

The blog is at [https://mobiusml.github.io/low-rank-llama2/](https://mobiusml.github.io/low-rank-llama2/)  
and code is at [https://github.com/mobiusml/low-rank-llama2/tree/main/code](https://github.com/mobiusml/low-rank-llama2/tree/main/code)
