# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# From https://github.com/Cornell-RelaxML/quip-sharp/

import torch


# load Hessian from files
def load_hessian(hessian_path, logger=None):
    if logger is None:
        print(f'load Hessian from {hessian_path}')
    else:
        logger.info(f'load Hessian from {hessian_path}')
    H_data = torch.load(f'{hessian_path}', weights_only=False)

    # convert H to sym matrix
    def flat_to_sym(V, N):
        A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
        idxs = torch.tril_indices(N, N, device=V.device)
        A[idxs.unbind()] = V
        A[idxs[1, :], idxs[0, :]] = V
        return A

    def regularize_H(H, n, sigma_reg):
        H.div_(torch.diag(H).mean())
        idx = torch.arange(n)
        H[idx, idx] += sigma_reg
        return H

    def basic_preprocess(H, mu, n):
        H.add_(mu[None, :] * mu[:, None])
        H = regularize_H(H, n, 1e-2)
        return H, mu

    H = flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    n = H_data['n']
    H, mu = basic_preprocess(H, mu, n)

    return H, mu


# load inverse Hessian from files
# TODO: reduce tensor size
def load_inv_hessian(inv_hessian_path, logger=None):
    if logger is None:
        print(f'load inv Hessian from {inv_hessian_path}')
    else:
        logger.info(f'load inv Hessian from {inv_hessian_path}')
    H_data = torch.load(f'{inv_hessian_path}', weights_only=False)

    inv_hessian = H_data['invH']
    perm = H_data['perm']
    zero_idx = H_data['zero_idx']

    return inv_hessian, perm, zero_idx
