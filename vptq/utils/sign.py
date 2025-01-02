# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

# [N, v] pack sign bits of vector to 16-bit integers
def pack_sign(data):
    signs = (torch.sign(data) < 0).to(torch.int8)
    
    pad_len = (16 - (signs.shape[1] % 16)) % 16
    if pad_len:
        signs = torch.nn.functional.pad(signs, (0, pad_len))
    
    signs = signs.view(-1, 16)
    packed_signs = torch.zeros(signs.size(0), dtype=torch.int16, device=signs.device)
    for i in range(signs.shape[1]):
        packed_signs |= (signs[:, i].to(torch.int16) << i)
        
    return packed_signs

# unpack sign bits of vector from 16-bit integers
def unpack_sign(packed_signs, vector_len):
    original_shape = packed_signs.shape
    packed_signs = packed_signs.reshape(-1)

    # Unpack bits for each integer
    unpacked = torch.zeros(packed_signs.shape[0], 16, dtype=torch.int8, device=packed_signs.device)
    for i in range(16):
        unpacked[:, i] = (packed_signs >> i) & 1
    
    # Convert to -1/+1 and trim to vector_len
    signs = (unpacked == 1).to(torch.int16) * -2 + 1
    signs = signs[:, :vector_len]
    
    # Restore original dimensions and add vector_len dimension
    new_shape = original_shape + (vector_len,)
    signs = signs.reshape(new_shape)
    
    return signs
    
if __name__ == '__main__':
    # Test with batch dimension
    test_cases = [
        torch.tensor([[-1, 2, -3, 4], [5, -6, 7, 8]]),
        torch.tensor([[-1, 1, -1, 1], [1, -1, 1, -1]]),
        torch.tensor([[-1, 1, -1, 1, 2, -2, 3, -3], [1, -1, 1, -1, -2, 2, -3, 3]]),
    ]
    
    for data in test_cases:
        print(f"\nOriginal data: ({data.shape, data.dtype})\n{data}")
        
        # Pack signs
        packed = pack_sign(data)
        print(f"Packed signs: ({packed.shape, packed.dtype})\n{packed}")
        
        unpacked = unpack_sign(packed, data.shape[1])
        print(f"Unpacked signs: ({unpacked.shape, unpacked.dtype})\n{unpacked}")
        
        # Verify
        original_signs = torch.sign(data)
        print(f"Original signs: {original_signs.shape}\n{original_signs}")
        print("Matches:", torch.all(unpacked == original_signs))