#!/bin/python
#-----------------------------------------------------------------------------
# File Name : hessian
# Author: Emre Neftci, Hin Wai Lui
#
# Creation Date : Sept 2. 2019
# Last Modified : May 3. 2021
#
# Copyright : (c) UC Regents, Emre Neftci, Hin Wai Lui
# Licence : GPLv2
# This file is part of PyHessian library https://github.com/amirgholami/PyHessian.
#-----------------------------------------------------------------------------
import torch
import numpy as np

from pyhessian.hessian import hessian
from pyhessian.utils import group_product, get_params_grad, hessian_vector_product


def group_mean(xs, axis=None):
    """
    the mean of each element of xs
    :param xs:
    :return:
    """
    if axis is not None:
        return [np.mean(x, axis=axis) for x in xs]
    else:
        return [np.mean(x) for x in xs]


class SSN_Hessian(hessian):
    """
    The class extends the hessian class form PyHessian and allows computation of layer wise SNN Hessian:
    """

    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader
        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for i, (inputs, targets) in enumerate(self.data):
            self.model.zero_grad()
            inputs = inputs.to(device)
            tmp_num_data = inputs.size(0)
            burnin = 50
            self.model.init(inputs, burnin=burnin)
            t_sample = inputs.shape[1]
            loss_tv = torch.tensor(0.).to(device)
            for t in (range(burnin, t_sample)):
                Sin_t = inputs[:, t]
                s_out, r_out, u_out =  self.model.step(Sin_t)
                for r in r_out:
                    loss_tv += self.criterion(r, targets[:, t].to(device))
                loss_tv.backward(create_graph=True)
                params, gradsH = get_params_grad(self.model)
                self.model.zero_grad()
                Hv = torch.autograd.grad(gradsH,
                                         params,
                                         grad_outputs=v,
                                         only_inputs=True,
                                         retain_graph=False)
                THv = [
                    THv1 + Hv1 * float(tmp_num_data) + 0.
                    for THv1, Hv1 in zip(THv, Hv)
                ]
                loss_tv = torch.tensor(0.).to(device)
            num_data += float(tmp_num_data)
        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """
        print("Computing Hessian Trace.")
        device = self.device
        trace_vhv = []
        trace = 0.
        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            _trace_vhv = [torch.sum(x * y).cpu().item() for (x, y) in zip(Hv, v)]
            if abs(np.mean(group_mean(trace_vhv)) - trace) / (trace + 1e-6) < tol:
                return group_mean(trace_vhv, axis=0)
            else:
                trace = np.mean(group_mean(trace_vhv))
        return group_mean(trace_vhv, axis=0)
