'''
Persistent entropy loss for point clouds.

The PyTorch and JAX implementations are independent, install the one you need:
    pip install "topoentropy[torch]"   ->  from topoentropy import entropy_loss
    pip install "topoentropy[jax]"     ->  from topoentropy.jax_loss import entropy_loss
'''

__all__ = ["entropy_loss"]


def __getattr__(name):
    # lazy, so that the JAX implementation can be used without PyTorch installed
    if name == "entropy_loss":
        from .loss_utils import entropy_loss
        return entropy_loss
    raise AttributeError(f"module 'topoentropy' has no attribute {name!r}")
