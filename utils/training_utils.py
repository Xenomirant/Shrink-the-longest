from collections import deque
import typing
import scipy
import numpy as np
import torch
from matplotlib import pyplot as plt
import ripser


def intrinsic_dimension(emb, debug=False, reduction_factor=5,):
    """
    emb: n x dim torch tensor
    """
    with torch.no_grad():
        eps = 1e-8
        embeddings = emb.to(torch.float64)
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        avg_len = (embeddings*embeddings).sum(dim=1).sqrt().mean()
        embeddings = embeddings / avg_len

        r1 = []
        r2 = []
        n = len(embeddings)
        for i in range(n):
            dsts = torch.nn.functional.pairwise_distance(
                embeddings[i, None, :],
                embeddings[None, :, :],
                eps=0
            )[0]
            dsts = torch.cat([dsts[:i], dsts[i+1:]])
            r1.append(torch.kthvalue(dsts, k=1)[0])
            r2.append(torch.kthvalue(dsts, k=2)[0])
        r1 = torch.tensor(r1).to(emb.device)
        r2 = torch.tensor(r2).to(emb.device)
        bad_cases = (r1 < eps)
        r1[bad_cases] = eps
        r2[bad_cases] = eps
        mu = r2 / r1
        mu[bad_cases] = -1
    
        mu, ind = torch.sort(mu)
        all_mu = mu.clone().cpu().detach()
        useless_items = int((mu <= 1+eps).sum()) 
        mu = mu[useless_items:]
        n = mu.shape[0]
        if debug:
            print('Removed points: ', useless_items)
            plt.plot(mu.cpu().detach().numpy())
            plt.show()

        f_emp = torch.arange(1+useless_items, n + 1 + useless_items, device=emb.device) / (n + useless_items)
        num_dots_to_use = min(n  // reduction_factor, n - 1)
        
        mu_log = torch.log(mu)[:num_dots_to_use]
        dist_log = -torch.log(1 - f_emp)[:num_dots_to_use]

        if debug:
            print('Regression points:', len(mu_log))
            plt.scatter(mu_log.cpu().detach().numpy(), dist_log.cpu().detach().numpy(), marker='.')
            plt.show()

        dim = float((mu_log*dist_log).sum() / (mu_log*mu_log).sum())

        if debug:
            print('Dim: ', dim)
    return float(dim) #, all_mu


# from Birdal et al, Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks (NIPS 2021)
def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=isRandom)
    return W[random_indices]


def calculate_ph_dim(W, min_points=150, max_points=800, point_jump=50,  
        h_dim=0, print_error=False):
    
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        diagrams = ripser(sample_W(W, n))['dgms']
        
        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append((d[:, 1] - d[:, 0]).sum())
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)
    
    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)


def compute_anisotropy(matrix, n_components: typing.Union[str, int] = 3, write_to_file: typing.Optional[str] = None) -> float:
    '''
    Computes anisotropy for a given matrix
    Params:
    matrix -- array for computation
    write_to_file -- whether to write to the specified filename; if not, returns as output
    '''
    res = scipy.linalg.svd(matrix, overwrite_a = True, full_matrices = False, compute_uv=False)
    if n_components == "all":
        anisotropy = res
    else:
        anisotropy = res[:n_components]**2 / np.sum(res**2)

    if write_to_file is None:
        return anisotropy

    with open(write_to_file, "a") as f:
        print(", ".join([str(val) for val in anisotropy]), end="\n", file = f)
    return None


def get_last_line(filename):
    try:
        with open(filename, 'r') as f:
            lastline = deque(f, 1)[0]
    except FileNotFoundError:
        lastline = "0.0"
    return lastline


def gradient_norm(model):
        grads = [
        param.grad.detach().flatten()
            for param in model.parameters()
                if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm
    