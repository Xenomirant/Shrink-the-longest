## Shrink the longest: improving latent space isotropy with simplicial geometry

#### Project structre

```
project/
├── utils/
│   ├── loss_utils.py
│   └── training_utils.py
├──model_finetuning.ipynb
└── README.md
```

__The primary algorithm implementation for separation of topological features from noise__ can be found in [loss_utils.py (prominient_features)](utils/loss_utils.py#L38)

#

__Entropy loss__ supports both entropy maximization of the distance distribution on persistant features found, and an additional term for l2 minimization within clusters.

To install as a python package:

```
pip install git+https://github.com/xenos/Shrink-the-longest.git@main
```

#### Model finetuning scrips and visualization are located in [model_finetuning.ipynb](model_finetuning.ipynb).


### Cite
```bibtex
@misc{kudriashov2025shrinklongestimprovinglatent,
      title={Shrink the longest: improving latent space isotropy with symplicial geometry}, 
      author={Sergei Kudriashov and Olesya Karpik and Eduard Klyshinsky},
      year={2025},
      eprint={2501.05502},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.05502}, 
}
```