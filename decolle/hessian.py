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
from pyhessian.utils import group_product, hessian_vector_product


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for name, param in model.named_parameters():
        if not param.requires_grad or 'bias' in name:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


class SSN_Hessian(hessian):
    """
    The class extends the hessian class form PyHessian and allows computation of layer wise SNN Hessian:
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """
        super().__init__(model, criterion, data, dataloader, cuda)
        # this step is used to extract the parameters from the model, using our modified function.
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader
        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for i, (inputs, targets) in enumerate(self.data):
            self.model.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            tmp_num_data = inputs.size(0)
            burnin = 50
            self.model.init(inputs, burnin=burnin)
            t_sample = inputs.shape[1]
            loss_mask = (targets.sum(2)>0).unsqueeze(2).float().to(device)
            for t in (range(burnin, t_sample)):
                Sin_t = inputs[:, t]
                s, r, u = self.model.step(Sin_t)
                loss_ = self.criterion(s, r, u, target=targets[:,t,:], mask = loss_mask[:,t,:], sum_ = False)
                sum(loss_).backward(create_graph=True)
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
            trace_vhv.append(_trace_vhv)
            if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
                return np.mean(trace_vhv, axis=0)
            else:
                trace = np.mean(trace_vhv)
        return np.mean(trace_vhv, axis=0)
