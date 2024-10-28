# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch


# reshape class for matrix to vectors
class reshape():

    def __init__(self, vector_len, enable_transpose=False):
        self.vector_len = vector_len
        self.enable_transpose = enable_transpose
        self.is_padded = False
        self.pad_cols = 0

    def matrix2vectors(self, data):
        if data is None:
            return None, None
        if self.enable_transpose:
            data = data.T
        data, self.is_padded, self.pad_cols = self.add_padding(data)
        self.padded_shape = data.shape
        sub_vectors = data.reshape(-1, self.vector_len)
        return sub_vectors, self.padded_shape

    def add_padding(self, data):
        '''
        Check if data need padding columns
        Returns (padded data, is_padded, pad_cols)
        '''
        remainder = data.shape[1] % self.vector_len
        if remainder != 0:
            padded_tensor = torch.zeros((data.shape[0], self.vector_len - remainder),
                                        dtype=data.dtype,
                                        device=data.device)
            return torch.cat((data, padded_tensor), dim=1), True, self.vector_len - remainder
        return data, False, 0

    def remove_padding(self, data):
        '''
        Remove padding
        '''
        if self.is_padded:
            if self.enable_transpose:
                data = data[:, :-self.pad_cols].T
            else:
                data = data[:, :-self.pad_cols]
        else:
            if self.enable_transpose:
                data = data.T
            else:
                data = data
        return data
