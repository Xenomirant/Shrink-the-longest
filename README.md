## Shrink the longest: improving latent space isotropy with simplicial geometry

#### Project structre

```
project/
├── topoentropy/
│   ├── loss_utils.py    # PyTorch implementation
│   └── jax_loss.py      # JAX implementation
├── tests/
├── utils/
│   └── training_utils.py
├──model_finetuning.ipynb
└── README.md
```

__The primary algorithm implementation for separation of topological features from noise__ can be found in [loss_utils.py (prominient_features)](topoentropy/loss_utils.py)

#

__Entropy loss__ supports both entropy maximization of the distance distribution on persistant features found, and an additional term for l2 maximization within clusters. The returned value is meant to be maximized, i.e. subtracted from the training loss.

To install as a python package, choose the PyTorch or the JAX implementation. They are independent and have the same interface:

```
pip install "topoentropy[torch] @ git+https://github.com/xenomirant/Shrink-the-longest.git@main"
pip install "topoentropy[jax] @ git+https://github.com/xenomirant/Shrink-the-longest.git@main"
```

Connected components (`max_dim=1`, the setting used in the paper) are computed on the device of the input, including GPU, with no further dependencies. Higher-order simplices (`max_dim > 1`) are computed on CPU and require Gudhi, installed with the additional `[higher]` extra.

```python
from topoentropy import entropy_loss            # PyTorch
# from topoentropy.jax_loss import entropy_loss # JAX, works under jit and grad

ent, n_features = entropy_loss(embeddings, max_dim=1)
loss = ce_loss - ent
```

Both implementations are tested to return the same values and gradients (`pytest tests`). On an H100 the loss with its backward pass takes about 2 ms for up to 256 points in PyTorch, see `benchmarks/`.

#### Model finetuning scrips and visualization are located in [model_finetuning.ipynb](model_finetuning.ipynb).


### Cite
```bibtex
@misc{kudriashov2025shrinklongestimprovinglatent,
      title={Shrink the longest: improving latent space isotropy with simplicial geometry}, 
      author={Sergei Kudriashov and Olesya Karpik and Eduard Klyshinsky},
      year={2025},
      eprint={2501.05502},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.05502}, 
}
```