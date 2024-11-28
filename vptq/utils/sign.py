# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

# pack sign bits of vector to 16-bit integers
def pack_sign(data):
    signs = (torch.sign(data) < 0).to(torch.int8)
    
    pad_len = (16 - (signs.numel() % 16)) % 16
    if pad_len:
        signs = torch.nn.functional.pad(signs, (0, pad_len))
    
    signs = signs.view(-1, 16)
    packed_signs = torch.zeros(signs.size(0), dtype=torch.int16, device=signs.device)
    for i in range(16):
        packed_signs |= (signs[:, i].to(torch.int16) << i)
    
    return packed_signs

# unpack sign bits of vector from 16-bit integers
def unpack_sign(packed_signs, original_size):
    unpacked = torch.zeros(packed_signs.size(0) * 16, dtype=torch.int8, device=packed_signs.device)
    for i in range(16):
        unpacked[i::16] = (packed_signs >> i) & 1
    signs = (unpacked == 1).to(torch.int16) * -2 + 1
    signs = signs[:original_size]
    return signs

if __name__ == '__main__':
    # Test with various sizes
    test_cases = [
        [-1, 2, -3, 4, 5, -6, 7, 8],
        [-1, 1] * 10,
        [1, -1] * 16,
    ]
    
    for data in test_cases:
        data = torch.tensor(data)
        print(f"\nOriginal data: ({data.shape, data.dtype}) {data}")
        
        # Pack signs
        packed = pack_sign(data)
        print(f"Packed signs: ({packed.shape, packed.dtype}) {packed}")
        
        # Unpack signs
        unpacked = unpack_sign(packed, data.numel())
        print(f"Unpacked signs: ({unpacked.shape, unpacked.dtype}) {unpacked}")
        
        # Verify
        original_signs = torch.sign(data)
        print(f"Original signs: ({original_signs.shape}) {original_signs}")
        print("Matches:", torch.all(unpacked == original_signs))
