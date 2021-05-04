#!/bin/python
#-----------------------------------------------------------------------------
# File Name : init_functions.py
# Author: Emre Neftci
#
# Creation Date : Fri 26 Feb 2021 11:48:40 AM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import torch
import numpy as np

from torch.nn import init


def init_LSUV(net, data_batch):
    '''
    Initialization inspired from Mishkin D and Matas J. All you need is a good init. arXiv:1511.06422 [cs],
February 2016.
    '''
    ##Initialize
    with torch.no_grad():
        net.init_parameters(data_batch)
        #def lsuv(net, data_batch):
        for l in net.LIF_layers:
            l.base_layer.bias.data *= 0
            init.orthogonal_(l.base_layer.weight)
        alldone = False
        while not alldone:
            alldone = True
            s,r,u = net.process_output(data_batch)
            for i in range(len(net)):
                v=np.var(u[i][-1].flatten())
                m=np.mean(u[i][-1].flatten())
                mus=np.mean(s[i][-1].flatten())
                print(i,v,m,mus)
                if np.abs(v-1)>.1:
                    net.LIF_layers[i].base_layer.weight.data /= np.sqrt(v)
                    done=False
                else:
                    done=True
                    
                if np.abs(m+.1)>.2:
                    net.LIF_layers[i].base_layer.bias.data -= .5*m
                    done=False
                else:
                    done=True
                alldone*=done
