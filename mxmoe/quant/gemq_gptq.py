"""GEMQ-compatible GPTQ weight quantizer, vendored for standalone MxMoE use.

This is the plain GPTQ path used by GEMQ.  Keeping a local copy lets the MxMoE
comparison use the same PTQ implementation without importing or modifying the
GEMQ repository at runtime.
"""

import math

import torch
import torch.nn as nn


class GPTQWeightQuantizer(nn.Module):
    def __init__(
        self,
        x,
        name="",
        nbits=4,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        mse=False,
    ):
        super().__init__()
        self.name = name
        self.nbits = nbits
        self.groupsize = groupsize
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.actorder = actorder
        self.static_groups = static_groups
        self.mse = mse

        self.rows = x.shape[0]
        self.columns = x.shape[1]
        self.W = x.clone()
        self.H = torch.zeros((self.columns, self.columns), device=x.device)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        bsz = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()
        if torch.isnan(inp).any():
            raise ValueError(self.nsamples, f"NaN detected in input to {self.name}.")
        self.H *= self.nsamples / (self.nsamples + bsz)
        self.nsamples += bsz
        inp = math.sqrt(2.0 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def find_params(self, x):
        max_int = 2**self.nbits - 1
        if max_int == 1:
            scales = torch.mean(torch.abs(x), dim=1, keepdim=True) * 2
            zeros = 0.5 * torch.ones_like(scales)
            return scales, zeros, max_int

        tmp = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        min_val = torch.minimum(x.amin(dim=1, keepdim=True), tmp)
        max_val = torch.maximum(x.amax(dim=1, keepdim=True), tmp)
        empty = (min_val == 0) & (max_val == 0)
        min_val[empty] = -1
        max_val[empty] = 1

        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = torch.round(-min_val / scales)

        if self.mse:
            grid = 100
            maxshrink = 0.8
            norm = 2.4
            best = torch.full(
                [x.shape[0]], float("inf"), device=x.device, dtype=x.dtype
            )
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                minv = p * min_val
                maxv = p * max_val
                tmp_scales = (maxv - minv) / max_int
                tmp_zeros = torch.round(-minv / tmp_scales)
                q = self.quantize_vector(x, tmp_scales, tmp_zeros, max_int)
                q = (q - tmp_zeros) * tmp_scales
                q -= x
                q.abs_()
                q.pow_(norm)
                err = torch.sum(q, 1)
                better = err < best
                if torch.any(better):
                    best[better] = err[better]
                    scales[better] = tmp_scales[better]
                    zeros[better] = tmp_zeros[better]
        return scales, zeros, max_int

    @staticmethod
    def quantize_vector(x, scales, zeros, max_int):
        if max_int == 1:
            return torch.where(x >= 0, 1, 0)
        return torch.clamp(torch.round(x / scales) + zeros, 0, max_int)

    def quantize(self):
        device = self.W.device
        if self.nbits == 0:
            num_groups = (
                self.W.numel() // self.groupsize
                if self.groupsize > 0
                else self.W.shape[0]
            )
            qweight = torch.zeros_like(self.W).reshape(num_groups, -1)
            scales = torch.ones(
                num_groups, 1, dtype=self.W.dtype, device=device
            )
            zeros = torch.zeros_like(scales)
            return qweight, scales, zeros

        weight = self.W.float()
        hessian = self.H
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        if self.static_groups:
            raise NotImplementedError("GEMQ-compatible GPTQ requires static_groups=False.")
        if self.actorder:
            raise NotImplementedError("GEMQ-compatible GPTQ requires actorder=False.")

        losses = torch.zeros_like(weight)
        qweight = torch.zeros_like(weight)
        damp = self.percdamp * torch.mean(torch.diag(hessian))
        diag = torch.arange(self.columns, device=device)
        hessian[diag, diag] += damp
        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hinv = torch.linalg.cholesky(hessian, upper=True)

        scales, zeros, max_int = self.find_params(weight)
        scales_list = []
        zeros_list = []

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1
            weight_block = weight[:, i1:i2].clone()
            q_block = torch.zeros_like(weight_block)
            err_block = torch.zeros_like(weight_block)
            loss_block = torch.zeros_like(weight_block)
            hinv_block = hinv[i1:i2, i1:i2]

            for i in range(count):
                w = weight_block[:, i]
                d = hinv_block[i, i]
                if self.groupsize > 0 and (i1 + i) % self.groupsize == 0:
                    scales, zeros, max_int = self.find_params(
                        weight[:, (i1 + i) : (i1 + i + self.groupsize)]
                    )
                    scales_list.append(scales)
                    zeros_list.append(zeros)

                q = self.quantize_vector(w.unsqueeze(1), scales, zeros, max_int)
                restored = ((q - zeros) * scales).flatten()
                q = q.flatten()
                q_block[:, i] = q
                loss_block[:, i] = (w - restored) ** 2 / d**2
                err = (w - restored) / d
                weight_block[:, i:] -= err.unsqueeze(1).matmul(
                    hinv_block[i, i:].unsqueeze(0)
                )
                err_block[:, i] = err

            qweight[:, i1:i2] = q_block
            losses[:, i1:i2] = loss_block / 2
            weight[:, i2:] -= err_block.matmul(hinv[i1:i2, i2:])

        if self.groupsize != -1:
            scales = torch.cat(scales_list, dim=1).to(self.W.dtype)
            zeros = torch.cat(zeros_list, dim=1).to(self.W.dtype)
            qweight = qweight.reshape(-1, self.groupsize)
            scales = scales.reshape(-1, 1)
            zeros = zeros.reshape(-1, 1)

        return (
            qweight.to(self.W.dtype),
            scales.to(self.W.dtype),
            zeros.to(self.W.dtype),
        )

    @staticmethod
    def dequantize(qweight, scales, zeros):
        return (qweight - zeros) * scales
