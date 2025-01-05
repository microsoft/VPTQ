# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import time

from numpy import int_
import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class VPTQ:

    def __init__(
        self,
        layer,
        quantizer,
        hessian,
        inv_hessian,
        perm,
        zero_idx,
        logger,
        collect_act=False,
        layer_name='',
        block_size=128,
        step=1,
        percdamp=.01,
        group_size=-1,
        group_num=-1,
        enable_perm=False,
        enable_norm=False,
        enable_sphere=False,
        norm_dim=0,
        enable_abs=False,
        debug=False
    ):
        # set layer
        # self.layer = layer

        # set quantizer
        self.quantizer = quantizer

        # vptq parameter
        self.block_size = block_size
        self.step = step
        self.percdamp = percdamp

        self.group_size = group_size
        self.group_num = group_num

        # set device
        self.dev = layer.weight.device
        # layer name
        self.layer_name = layer_name

        # save weight
        # self.weight = self.layer.weight.data.to(self.dev)
        # self.qweight = torch.zeros_like(self.weight)
        # self.hessian = hessian.to(self.dev)

        # preprocess
        self.layer = layer.to('cpu')
        self.weight = self.layer.weight.data.to('cpu')
        self.qweight = torch.zeros_like(self.weight).to('cpu')
        self.hessian = hessian.to('cpu')

        if inv_hessian is not None:
            self.inv_hessian = inv_hessian.to('cpu')
        else:
            self.inv_hessian = None
        if perm is not None:
            self.perm = perm.to('cpu')
        else:
            self.perm = None
        if zero_idx is not None:
            self.zero_idx = zero_idx.to('cpu')
        else:
            self.zero_idx = None

        if isinstance(self.layer, nn.Conv2d):
            self.weight = self.weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            self.weight = self.weight.t()

        # out_features
        self.rows = self.weight.shape[0]
        # in_features
        self.columns = self.weight.shape[1]

        # nsamples
        self.nsamples = 0

        # hessian matrix
        # self.hessian = torch.zeros(
        #     (self.columns, self.columns), device=self.dev)

        # collect activation
        self.collect_act = collect_act
        self.act = None

        # permute
        self.quantizer.enable_perm = enable_perm
        # self.quantizer.perm = None

        # weight norm
        self.enable_norm = enable_norm
        self.norm_dim = norm_dim
        self.enable_sphere = enable_sphere
        
        self.enable_abs = enable_abs
        # self.quantizer.weight_scale = None
        # self.quantizer.weight_bias = None

        # debug flag
        self.debug = debug
        self.logger = logger

    # vector quant
    def fast_vector_quant(self, init_centroids=True):
        self.init_centroids = init_centroids

        # step 0: preprocess weight and hessian
        weight = self.weight.clone().float().to(self.dev)
        hessian = self.hessian.clone().to(self.dev)
        inv_hessian = self.inv_hessian.clone().to('cpu')

        if self.enable_norm:
            # norm weight for quantization
            # self.quantizer.weight_scale = torch.linalg.norm(
            #     weight, dim=self.norm_dim)
            # self.quantizer.weight_bias = torch.mean(
            #     weight, dim=self.norm_dim)
            # if self.debug:
            #     self.logger.info(
            #         f'enabling norm dim {self.norm_dim}, '
            #         f'layer_name:{self.layer_name}, '
            #         f'scale:{self.quantizer.weight_scale.shape}, '
            #         f'bias:{self.quantizer.weight_bias.shape}')
            self.quantizer.init_norm(weight)

            weight = (weight - self.quantizer.weight_bias.unsqueeze(self.norm_dim)) / \
                self.quantizer.weight_scale.unsqueeze(self.norm_dim)
                    
        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()

        # set group num and size
        self.outlier_size, self.group_size, self.group_num = \
            self.quantizer.get_group_setting(weight)

        # print(f'group_size: {self.group_size}, group_num: {self.group_num}')

        # if not self.quantizer.ready():
        #     self.quantizer.find_params(W, weight=True)
        # del self.H

        if self.quantizer.kmeans_mode == 'hessian':
            kmeans_weight = torch.diag(hessian).clone().unsqueeze(0).repeat(weight.shape[0], 1)
        else:
            kmeans_weight = None

        # set dead diagonal after kmeans_weight
        if self.zero_idx is None:
            self.zero_idx = torch.diag(hessian) == 0
        hessian[self.zero_idx, self.zero_idx] = 1
        weight[:, self.zero_idx] = 0

        if self.debug:
            self.logger.info(
                f'kmeans_mode: {self.quantizer.kmeans_mode}, '
                f'enable_perm: {self.quantizer.enable_perm}, '
                f'enable_sphere: {self.quantizer.enable_sphere}, '
                f'enable_norm: {self.quantizer.enable_norm}, '
                f'enable_abs: {self.quantizer.enable_abs}'
            )

        # permute weight and hessian
        if self.quantizer.enable_perm:
            self.quantizer.init_perm(hessian, self.perm)
            # reorder weight and H
            weight = weight[:, self.quantizer.perm]
            hessian = hessian[self.quantizer.perm][:, self.quantizer.perm]

            # reorder kmeans_weight
            if self.quantizer.kmeans_mode in ['hessian'] \
                    and kmeans_weight is not None:
                kmeans_weight = kmeans_weight[:, self.quantizer.perm]
            else:
                kmeans_weight = None
        else:
            # reverse perm hesisan
            if self.perm is not None:
                inv_perm = torch.argsort(self.perm)
                inv_hessian = inv_hessian[inv_perm][:, inv_perm]
            self.quantizer.perm = torch.arange(weight.shape[1])

        # save gpu memory
        weight = weight.to('cpu')
        hessian = hessian.to('cpu')
        inv_hessian = inv_hessian.to('cpu')
        # end of weight and hessian preprocess

        # step 1: init centroids ###
        if self.init_centroids is True:
            # clone weight and hessian, vptq will modify them in-place
            _weight = weight.clone().to(self.dev)
            _hessian = hessian.clone().to(self.dev)
            
            tick = time.time() if self.debug else None

            # run k-means, init centroids and get quantized data by k-means
            qweight_init = self.quantizer.init_centroids_indices(data=_weight, weights=kmeans_weight)
            
            if self.debug:
                self.logger.info(f'{self.layer_name} 1st kmeans time: {time.time() - tick}')
                self.logger.info(
                    f'{self.layer_name} qweight init shape: {qweight_init.shape}, '
                    f'weight shape: {weight.shape}'
                )

                error_sum, sum, error_norm = self.get_error(weight, qweight_init, hessian)
                self.logger.info(
                    f'{self.layer_name} proxy error before VPTQ: '
                    f'{error_sum.item()}, {sum.item()}, {error_norm.item()}'
                )

        # step 2: VPTQ with initialized centroids ###
        _weight = weight.clone().to(self.dev)
        _hessian = hessian.clone().to(self.dev)
        _inv_hessian = inv_hessian.clone().to(self.dev)
        tick = time.time() if self.debug else None

        # first round vptq
        qweight, qerror = self.vptq(_weight, _hessian, inv_hessian=_inv_hessian)

        torch.cuda.synchronize()
        del _weight
        del _hessian
        del _inv_hessian
        torch.cuda.empty_cache()

        if self.debug:
            error_sum, sum, norm_error = self.get_error(weight, qweight, hessian)
            self.logger.info(f'{self.layer_name} 1st error time: {time.time() - tick}')
            self.logger.info(
                f'{self.layer_name} proxy error after VPTQ: {error_sum.item()}, '
                f'{sum.item()}, {norm_error.item()}'
            )
            # debug 
            # self.logger.info(f'qerror^2: {torch.mean(qerror ** 2).item()}')
            # torch.save(qweight, f'{self.layer_name}_qweight.pt')

        # step 3: residual quantization
        if self.quantizer.num_res_centroids[1] > 1:  # (100-N)%
            if self.init_centroids is True:
                # quant residual
                tick = time.time() if self.debug else None

                # step 3.1: init residual quantization centroids
                qweight_residual = self.quantizer.init_res_centroids_indices(qerror, kmeans_weight)

                # torch.save(Q_residual, f'Q_residual_{self.layer_name}.pt')

                if self.debug:
                    self.logger.info(f'{self.layer_name} residual time: {time.time() - tick}')

            _weight = weight.clone().to(self.dev)
            _hessian = hessian.clone().to(self.dev)
            _inv_hessian = inv_hessian.clone().to(self.dev)

            # step 3.2: VPTQ with initialzed residual centroids
            self.quantizer.clear_indices()
            tick = time.time()
            qweight, qerror = self.vptq(_weight, _hessian, enable_residual=True, inv_hessian=_inv_hessian)

            if self.debug:
                self.logger.info(f'{self.layer_name} 2ed gptq time: {time.time() - tick}')

            tick = time.time()

            del _weight
            del _hessian
            del _inv_hessian
            torch.cuda.empty_cache()

            if self.debug:
                self.logger.info(f'{self.layer_name} 2ed error time: {time.time() - tick}')
                error_sum, sum, norm_error = self.get_error(weight, qweight, hessian)
                self.logger.info(
                    f'{self.layer_name} proxy error after residual VPTQ: '
                    f'{error_sum.item()}, {sum.item()}, {norm_error.item()}'
                )

        # self.quantizer.save(qweight)

        if self.quantizer.enable_perm:
            inv_perm = torch.argsort(self.quantizer.perm)
            qweight = qweight[:, inv_perm]
            # self.quantizer.perm = self.quantizer.perm.cpu().numpy()

        if isinstance(self.layer, transformers.Conv1D):
            qweight = qweight.t()

        # reshape back to original shape
        qweight = qweight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.enable_norm:
            qweight = qweight * self.quantizer.weight_scale.unsqueeze(self.norm_dim) + \
                self.quantizer.weight_bias.unsqueeze(self.norm_dim)

        self.qweight = qweight

        # post process
        self.layer = self.layer.to(self.dev)
        self.weight = self.weight.to(self.dev)
        self.qweight = self.qweight.to(self.dev)
        torch.cuda.empty_cache()

    # vptq algorithm, we do not permute weight and hessian here
    def vptq(self, weight, hessian, enable_residual=False, inv_hessian=None):
        # force set step=1 if transposed
        self.step = 1 if self.quantizer.enable_transpose else self.step

        # error = torch.zeros_like(weight)
        qweight = torch.zeros_like(weight)

        # gptq error
        qerror = torch.zeros_like(weight)

        if inv_hessian is None:
            damp = self.percdamp * torch.mean(torch.diag(hessian))
            diag = torch.arange(self.columns, device=self.dev)
            hessian[diag, diag] += damp

            # inverse Hessian
            hessian = torch.linalg.cholesky(hessian)
            hessian = torch.cholesky_inverse(hessian)

            # follow gptq methods
            # upper=True: whether to return an upper triangular matrix
            # compute all information needed from H-1 upfront

            hessian = torch.linalg.cholesky(hessian, upper=True)
            inv_hessian = hessian
        else:
            inv_hessian = inv_hessian
            assert inv_hessian is not None

        # select weight[i, j] to quantize
        for i in range(0, self.columns, self.block_size):
            j = min(i + self.block_size, self.columns)
            size = j - i

            block_weight = weight[:, i:j].clone()
            block_qweight = torch.zeros_like(block_weight)
            # self.logger.info(f'Q1.shape:{Q1.shape}')
            block_error = torch.zeros_like(block_qweight)
            # Losses1 = torch.zeros_like(weight)
            # block of Hessian
            block_inv_hessian = inv_hessian[i:j, i:j]

            for k in range(0, size, self.step):
                tile_weight = block_weight[:, k:k + self.step]
                tile_inv_hessian = block_inv_hessian[k:k + self.step, k:k + self.step]

                # self.logger.info(f'tile_weight:{tile_weight.shape}, tile_inv_hessian:{tile_inv_hessian.shape}')
                if enable_residual:
                    tile_qweight = self.quantizer.quantize_residual_vector(tile_weight, i + k)
                else:
                    tile_qweight = self.quantizer.quantize_vector(tile_weight, i + k)
                # self.logger.info(f'tile_qweight:{tile_qweight.shape}')

                # ?
                tile_qweight = tile_qweight.reshape(-1, self.step)

                tile_inv_hessian = torch.cholesky_inverse(torch.linalg.cholesky(tile_inv_hessian))

                # self.logger.info(f'block_qweight:{block_qweight.shape}, tile_qweight:{tile_qweight.shape}')
                # self.logger.info(f'block_qweight[:, k:k+self.step]: {block_qweight[:, k:k+self.step].shape}')

                # update quantized block qweight
                # norm_block_qweight = block_qweight * S + B
                block_qweight[:, k:k + self.step] = tile_qweight

                # (drow,step)*(step,step)=(drow,step)
                # [(norm_tile_weight - norm_tile_qweight) * S + B] * H^-1
                tile_error = (tile_weight - tile_qweight)
                
                # if self.enable_norm:
                #     tile_error = tile_error * self.quantizer.weight_scale.unsqueeze(self.norm_dim) + \
                #         self.quantizer.weight_bias.unsqueeze(self.norm_dim)
                
                tile_error = tile_error.matmul(tile_inv_hessian)
                # Losses1[:,i:i+step] =  err1.matmul((w-q).T())
                # [(norm_tile_weight - norm_tile_qweight) * S + B] * H^-1 * H^-1 * (w-q).T()

                inv_tile_error = tile_error.matmul(block_inv_hessian[k:k + self.step, k + self.step:])
                
                # if self.enable_norm:
                #     inv_tile_error = (inv_tile_error - self.quantizer.weight_bias.unsqueeze(self.norm_dim)) / \
                #         self.quantizer.weight_scale.unsqueeze(self.norm_dim)
                #     tile_error = (tile_error - self.quantizer.weight_bias.unsqueeze(self.norm_dim)) / \
                #         self.quantizer.weight_scale.unsqueeze(self.norm_dim)
                
                block_weight[:, k + self.step:] -= inv_tile_error 
                block_error[:, k:k + self.step] = tile_error

                # block_weight[:, k + self.step:] -= tile_error.matmul(block_inv_hessian[k:k + self.step, k + self.step:])
                # block_error[:, k:k + self.step] = tile_error

            qweight[:, i:j] = block_qweight
            # Losses[:, i1:i2] = Losses1 / 2

            # copy gptq error from Err1
            qerror[:, i:j] = (block_weight - block_qweight).clone()

            # update remaining full-preicision weight
            weight[:, j:] -= block_error.matmul(inv_hessian[i:j, j:])

            # self.logger.info('Quant Error(w-q):',W[:,i1:i2]-Q[:,i1:i2])
            # self.logger.info('Err1:',Err1)
            # self.logger.info('delta weight:', Err1.matmul(Hinv[i1:i2, i2:]) )
            # self.layer.weight.data[:, :j] = qweight[:, :j]
            # self.layer.weight.data[:, j:] = weight[:, j:]
        # self.logger.info(f'{self.layer_name} error: {i1}:{i2} {torch.sum((self.layer(self.inp1) - self.out1).double() ** 2).item()}')
        return qweight, qerror
    
    @torch.no_grad()
    def get_error(self, weight, qweight, hessian):

        def _matrix_multiply_with_blocks(A, B, hessian, block_size=64, dev='cuda'):
            m_dim = A.shape[0]
            k_dim = A.shape[1]
            n_dim = B.shape[1]
            if m_dim >= 16384 and k_dim >= 16384:
                # if m_dim >= 16 and k_dim >= 16:
                result = torch.zeros((m_dim, n_dim), device=dev, dtype=A.dtype)
                for i in range(0, m_dim, block_size):
                    i_end = min(i + block_size, m_dim)
                    for j in range(0, n_dim, block_size):
                        j_end = min(j + block_size, n_dim)
                        result[i:i_end, j:j_end] += A[i:i_end, :].to(dev) @ B[:, j:j_end].to(dev)
                        result[i:i_end, j:j_end] = result[i:i_end, j:j_end] * hessian[i:i_end, j:j_end]
            else:
                result = A.to(dev) @ B.to(dev) * hessian
            result = result.to(dev)
            return result

        # weight_mean = torch.mean(weight.T @ weight * hessian)
        # error_mean = torch.mean(error.T @ error * hessian)
        weight = weight.to(qweight.device)
        
        if self.enable_norm:
            scaled_weight = weight * self.quantizer.weight_scale.unsqueeze(self.norm_dim) + \
                self.quantizer.weight_bias.unsqueeze(self.norm_dim)
        else:
            scaled_weight = weight
        
        hessian = hessian.to(qweight.device)
        wTw_hessian = _matrix_multiply_with_blocks(scaled_weight.T, scaled_weight, hessian, block_size=512, dev=qweight.device)
        weight_mean = torch.mean(wTw_hessian.to(qweight.device))
        # weight_mean = torch.mean(wTw * hessian)
        del wTw_hessian
        torch.cuda.empty_cache()
        if self.enable_norm:
            scaled_qweight = qweight * self.quantizer.weight_scale.unsqueeze(self.norm_dim) + \
                self.quantizer.weight_bias.unsqueeze(self.norm_dim)
        else:
            scaled_qweight = qweight
        
        error = scaled_qweight - scaled_weight
        
        eTe_hessian = _matrix_multiply_with_blocks(error.T, error, hessian, block_size=512, dev=qweight.device)
        error_mean = torch.mean(eTe_hessian.to(qweight.device))
        del eTe_hessian
        torch.cuda.empty_cache()
        error_norm = error_mean / weight_mean

        return error_mean, weight_mean, error_norm
