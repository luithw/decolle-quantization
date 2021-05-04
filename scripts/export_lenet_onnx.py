#!/bin/python
#-----------------------------------------------------------------------------
# File Name : export_lenet_decolle_onnx
# Author: Tim Lui (hwlui@uci.edu)
# Modified from train_lenet_decolle
#
# Creation Date : Oct 7. 2020
# Last Modified :
#
# Copyright : (c) UC Regents, NMI
# Licence : GPLv2
#-----------------------------------------------------------------------------
from collections import namedtuple
from itertools import chain
import os

from decolle.lenet_decolle_model import LenetDECOLLE
from decolle import lenet_decolle_model
from decolle.base_model import StatelessWrapper
from decolle.utils import parse_args, load_model_from_checkpoint, prepare_experiment, extract_net_states

import numpy as np
import torch
import importlib
import onnx
from onnx import numpy_helper
import onnxruntime

torch.manual_seed(0)
np.random.seed(0)

np.set_printoptions(precision=4)
args = parse_args('parameters/params.yml')
assert args.resume_from is not None, "Please provided a trained lenet directory to be loaded."
args.no_save = True
params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args = args)
dataset = importlib.import_module(params['dataset'])

## Load Data
gen_train, gen_test = dataset.create_dataloader(chunk_size_train=params['chunk_size_train'],
                                  chunk_size_test=params['chunk_size_test'],
                                  batch_size=params['batch_size'],
                                  dt=params['deltat'],
                                  num_workers=params['num_dl_workers'])


# ## Create Model, Optimizer and Loss
def create_and_resume_model():
    net = LenetDECOLLE(out_channels=params['out_channels'],
                       Nhid=params['Nhid'],
                       Mhid=params['Mhid'],
                       kernel_size=params['kernel_size'],
                       stride=params['stride'],
                       pool_size=params['pool_size'],
                       input_shape=params['input_shape'],
                       alpha=params['alpha'],
                       alpharp=params['alpharp'],
                       beta=params['beta'],
                       num_conv_layers=params['num_conv_layers'],
                       num_mlp_layers=params['num_mlp_layers'],
                       lc_ampl=params['lc_ampl'],
                       lif_layer_type=getattr(lenet_decolle_model, params['lif_layer_type']),
                       method=params['learning_method'],
                       with_output_layer=True,
                       need_bias=params['need_bias']).to(args.device)

    print("Checkpoint directory " + dirs['checkpoint_dir'])
    opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    _ = load_model_from_checkpoint(dirs['checkpoint_dir'], net, opt, device=args.device)
    print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))
    return net, opt

net, opt = create_and_resume_model()
net.eval()

##Initialize
data_batch, target_batch = next(iter(gen_train))
data_batch = torch.Tensor(data_batch).to(args.device)

print('\n------Starting exporting model with {} layers to ONNX format-------'.format(len(net)))
net_inputs = data_batch[:, params['burnin_steps'], :, :]
s, r, u = net.step(net_inputs)   # Initialise the LIF layer states
torch_external_states = extract_net_states(net)
stateless_net = StatelessWrapper(net)

input_names = ['i_s'] + ["i_state_%s_%i" % (name, l) for l, layer in enumerate(torch_external_states) for name in layer._fields]
output_names = list(chain(*[["o_s_%i" % l, "o_r_%i" % l, "o_u_%i" % l] for l in range(len(net))])) + ["o_state_%s_%i" % (name, l) for l, layer in enumerate(torch_external_states) for name in layer._fields]
all_names = output_names + input_names
dynamic_axes = {name: {0: 'batch_size'} for name in all_names}

# Export the model
onnx_file = os.path.join(args.resume_from, "lenet.onnx")
torch.onnx.export(stateless_net,  # model being run
                  (net_inputs, torch_external_states),  # model input (or a tuple for multiple inputs)
                  os.path.join(args.resume_from, "lenet.onnx"),  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=12,  # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names=input_names,  # the model's input names
                  output_names=output_names,  # the model's output names
                  dynamic_axes=dynamic_axes,  # variable lenght axes
                  )

# Load ONNX model
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
initializers = onnx_model.graph.initializer
for initializer in initializers:
    W = numpy_helper.to_array(initializer)
    print("%s: %s" % (initializer.name, W.shape))

ort_session = onnxruntime.InferenceSession(onnx_file)

# ## Create the clean models again
net, opt = create_and_resume_model()
net.eval()
stateless_net = StatelessWrapper(net)

stateful_net, opt = create_and_resume_model()
stateful_net.eval()
onnx_external_states = {}

# Check if the onnx runtime output and the torch output is the same.
for t in range(params['burnin_steps'], params['burnin_steps'] + 10):
    print("\n\n================Comparing time step: %i================\n" % t)

    # Compute the torch model output
    net_inputs = data_batch[:, t, :, :]
    stateless_torch_outs, torch_external_states = stateless_net.step(net_inputs, torch_external_states)
    stateful_torch_outs = stateful_net.step(net_inputs)

    # Compute ONNX Runtime output
    ort_inputs = {}
    for ort_in_node in ort_session.get_inputs():
        if ort_in_node.name == "i_s":
            ort_inputs["i_s"] = net_inputs.cpu().numpy()
        else:
            ort_out_node_name = ort_in_node.name.replace('i_', 'o_')
            if ort_out_node_name in onnx_external_states.keys():
                ort_inputs[ort_in_node.name] = onnx_external_states[ort_out_node_name]
            else:
                ort_inputs[ort_in_node.name] = np.zeros([net_inputs.shape[0]] + ort_in_node.shape[1:], dtype=np.float32)
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_external_states = {}
    for ort_out_node, ort_out in zip(ort_session.get_outputs(), ort_outs):
        if "state" in ort_out_node.name:
            onnx_external_states[ort_out_node.name] = ort_out
    ort_outs = ort_outs[: 3 * len(net)]

    # compare ONNX Runtime and PyTorch results
    layer_names = ["s_%i" %i for i in range(len(net)) ] + ["r_%i" %i for i in range(len(net)) ] + ["u_%i" %i for i in range(len(net)) ]
    for i, (layer_name, stateful_torch, stateless_torch, ort) in enumerate(zip(layer_names, chain(*stateful_torch_outs), chain(*stateless_torch_outs), ort_outs)):
        torch_stateless_allclose = np.allclose(stateful_torch.detach().cpu().numpy(), stateless_torch.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)
        ort_stateful_al_close = np.allclose(stateful_torch.detach().cpu().numpy(), ort, rtol=1e-03, atol=1e-05)
        ort_stateless_allclose = np.allclose(stateless_torch.detach().cpu().numpy(), ort, rtol=1e-03, atol=1e-05)
        print("%s output %i passed, torch_stateless_allclose: %r, ort_stateful_al_close: %r, ort_stateless_allclose: %r" %
              (layer_name, i, torch_stateless_allclose, ort_stateful_al_close, ort_stateless_allclose))

        if i in [0, 4]:
            stateful_torch_np = stateful_torch.detach().cpu().numpy()
            sample_ix = np.array([np.random.choice(shape, size=1, replace=False) for shape in stateful_torch_np.shape]).squeeze()
            torch_sample = stateful_torch_np[sample_ix]
            ort_sample = ort[sample_ix]
            print("%s Sample %r torch: %r, orts: %r" % (layer_name, sample_ix, torch_sample, ort_sample))

        if not ort_stateful_al_close:
            stateful_torch_np = stateful_torch.detach().cpu().numpy()
            ix = np.array(np.where(stateful_torch_np != ort)).T
            sample_ix = tuple(i for i in ix[0])
            torch_sample = stateful_torch_np[sample_ix]
            ort_sample = ort[sample_ix]
            print("%s Diverging sample %r torch: %r, orts: %r" % (layer_name, sample_ix, torch_sample, ort_sample))
