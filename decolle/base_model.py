#!/bin/python
#-----------------------------------------------------------------------------
# File Name : base_model
# Author: Emre Neftci, Hin Wai Lui
#
# Creation Date : Sept 2. 2019
# Last Modified : May 3. 2021
#
# Copyright : (c) UC Regents, Emre Neftci, Hin Wai Lui
# Licence : GPLv2
#-----------------------------------------------------------------------------
import torch.nn as nn
import torch
from torch.onnx.symbolic_helper import parse_args
import numpy as np
from itertools import chain
from collections import namedtuple, OrderedDict
import warnings
from decolle.utils import get_output_shape, fixed_quantizers

dtype = torch.float32
sigmoid = nn.Sigmoid()
relu = nn.ReLU()


## from snntorch
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        #return input_/(1+slope * torch.abs(input_))
        return  (input_>0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        #10 here is the slope
        return grad_input / (10 * torch.abs(input_) + 1.0) ** 2

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    @parse_args('v')
    def symbolic(g, x):
        # See https://pytorch.org/docs/stable/onnx.html#adding-support-for-operators for adding custom operators
        # See https://github.com/onnx/onnx/blob/master/docs/Operators.md for a list of operatorss

        s = g.op('Greater', x, torch.zeros([1]))
        # TODO: Look up for the correct onnx dtype constant
        # https://github.com/onnx/onnx/blob/2cee65c7eaa4b4fa8f6e23ef7e4b546395b9b7f1/onnx/onnx.in.proto#L460
        # https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html
        s = g.op('Cast', s, to_i=1)
        return s

    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x > 0).type(x.dtype)

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input
    
class SigmoidStep(torch.autograd.Function):
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >=0).type(x.dtype)

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        res = torch.sigmoid(input)
        return res*(1-res)*grad_output

smooth_step = SmoothStep().apply
smooth_sigmoid = SigmoidStep().apply
fast_sigmoid = FastSigmoid.apply

class LinearFAFunction(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class FALinear(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, bias=True):
        super(FALinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        # weight has transposed form; more efficient (so i heard) (transposed at forward pass)
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # fixed random weight and bias for FA backward pass
        # does not need gradient
        self.weight_fa = torch.nn.Parameter(torch.FloatTensor(output_features, input_features), requires_grad=False)

        # weight initialization
        #torch.nn.init.kaiming_uniform(self.weight)
        #torch.nn.init.kaiming_uniform(self.weight_fa)
        #torch.nn.init.constant(self.bias, 1)
        # does not need gradient
        self.weight_fa = torch.nn.Parameter(torch.FloatTensor(output_features, input_features), requires_grad=False)

        # weight initialization
        #torch.nn.init.kaiming_uniform(self.weight)
        #torch.nn.init.kaiming_uniform(self.weight_fa)
        #torch.nn.init.constant(self.bias, 1)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)

def state_detach(state):
    for s in state:
        s.detach_()

class BaseLIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    sg_function = smooth_step

    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True, quantization=False, precision=None):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super(BaseLIFLayer, self).__init__()
        self.base_layer = layer
        self.deltat = deltat
        #self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.beta = torch.tensor(beta, requires_grad=False)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.do_detach = do_detach
        if quantization:
            print("LIF layer using quantizer with precision: %i" % precision)
            self.quantizer = fixed_quantizers[precision]
        else:
            self.quantizer = None

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'):
            n = layer.in_channels
            for k in layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if layer.bias is not None: 
                layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled data type, not resetting parameters')

    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''

        if type(layer) == torch.nn.Sequential:
            # layer is changed to a Sequential module by the quantisation package
            layer = layer[0]
        
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if type(layer) == torch.nn.Sequential:
            layer = layer[0]

        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        if type(self.base_layer) == torch.nn.Sequential:
            device = self.base_layer[0].weight.device
        else:
            device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self, Sin_a):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + self.tau_s * Sin_t #Wrong dynamics, kept for backward compatibility
        P = self.alpha * state.P + self.tau_m * state.Q #Wrong dynamics, kept for backward compatibility  
        R = self.alpharp * state.R - state.S * self.wrp
        if self.quantizer is not None:
            P = self.quantizer(P)
            Q = self.quantizer(Q)
            R = self.quantizer(R)
        U = self.base_layer(P) + R
        if self.quantizer is not None:
            U = self.quantizer(U)
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features
    
    def get_device(self):
        if type(self.base_layer) == torch.nn.Sequential:
            # layer is changed to a Sequential module by the quantisation package
            device = self.base_layer[0].weight.device
        else:
            device = self.base_layer.weight.device
        return device


class LIFLayer(BaseLIFLayer):
    sg_function  = FastSigmoid.apply

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + (1-self.beta)*Sin_t
        P = self.alpha * state.P + (1-self.alpha)*state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) #+ R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def init_parameters(self, Sin_t, *args, **kwargs):
        self.reset_parameters(self.base_layer, *args, **kwargs)
    
    def reset_parameters(self, layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'):
            layer.bias.data = layer.bias.data*((1-self.alpha)*(1-self.beta))
            layer.weight.data[:] *= 1
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:] *= 5e-2
            layer.bias.data[:] = layer.bias.data[:]*((1-self.alpha)*(1-self.beta))
        else:
            warnings.warn('Unhandled data type, not resetting parameters')


class SimpleLIFLayer(BaseLIFLayer):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'U', 'S', 't'])

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        if type(self.base_layer) == torch.nn.Sequential:
            # layer is changed to a Sequential module by the quantisation package
            device = self.base_layer[0].weight.device
        else:
            device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),    # Still keep Q for comaptability
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      U=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      t=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        P = self.alpha * state.P + Sin_t
        R = self.alpha * state.R + self.alpha * state.U * state.S

        # Add the constant bias current injection in
        if type(self.base_layer) == torch.nn.Sequential:
            bias = self.base_layer[0].bias
        else:
            bias = self.base_layer.bias
        view_shape = (1, bias.shape[0], 1, 1)
        if len(R.shape) == 2:
            view_shape = (1, bias.shape[0])
        bias = bias.view(view_shape).expand_as(R)
        R -= self.alpha * (state.t > 0).type(R.dtype) * bias  # Only add the bias when t > 0

        if self.quantizer is not None:
            P = self.quantizer(P)
            R = self.quantizer(R)
        U = self.base_layer(P) - R
        if self.quantizer is not None:
            U = self.quantizer(U)
        S = self.sg_function(U)
        t = state.t + 1

        self.state = self.NeuronState(P=P, R=R, Q=state.Q, U=U.detach(), S=S, t=t.detach())
        if self.do_detach:
            state_detach(self.state)
        return S, U


class LIFLayerNonorm(LIFLayer):
    sg_function  = smooth_step

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + Sin_t
        P = self.alpha * state.P + state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def reset_parameters(self, layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'): #its a convolution
            n = layer.in_channels
            for k in layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if layer.bias is not None:
                layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled data type, not resetting parameters')

class LIFLayerVariableTau(LIFLayer):
    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, random_tau=True, do_detach=True):
        super(LIFLayerVariableTau, self).__init__(layer, alpha, alpharp, wrp, beta, deltat)
        self.random_tau = random_tau
        self.alpha_mean = self.alpha
        self.beta_mean = self.beta
        self.do_detach = do_detach
        
    def randomize_tau(self, im_size, tau, std__mean = .25, tau_min = 5., tau_max = 200.):
        '''
        Returns a random (normally distributed) temporal constant of size im_size computed as
        `1 / Dt*tau where Dt is the temporal window, and tau is a random value expressed in microseconds
        between low and high.
        :param im_size: input shape
        :param mean__std: mean to standard deviation
        :return: 1/Dt*tau
        '''
        tau_v = torch.empty(im_size)
        tau_v.normal_(1, std__mean)
        tau_v.data[:] *= tau 
        tau_v[tau_v<tau_min]=tau_min
        tau_v[tau_v>=tau_max]=tau_max
        #tau = np.broadcast_to(tau, (im_size[0], im_size[1], channels)).transpose(2, 0, 1)
        return torch.Tensor(1 - 1. / tau_v)    
    
    def init_parameters(self, Sin_t):
        device = self.get_device()
        input_shape = list(Sin_t.shape)
        if self.random_tau:
            tau_m = 1./(1-self.alpha_mean)
            tau_s = 1./(1-self.beta_mean)
            self.alpha = self.randomize_tau(input_shape[1:], tau_m).to(device)
            self.beta  = self.randomize_tau(input_shape[1:], tau_s).to(device)
        else:
            tau_m = 1./(1-self.alpha_mean)
            tau_s = 1./(1-self.beta_mean)
            self.alpha = torch.ones(input_shape[1:]).to(device)*self.alpha_mean.to(device)
            self.beta  = torch.ones(input_shape[1:]).to(device)*self.beta_mean.to(device)
        self.alpha = self.alpha.view(Sin_t.shape[1:])
        self.beta  = self.beta.view(Sin_t.shape[1:])
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad = False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad = False)
        self.reset_parameters(self.base_layer)

class DECOLLEBase(nn.Module):
    requires_init = True
    output_statenames = OrderedDict(zip(['s', 'r', 'u'],[0, 1, 2]))
    def __init__(self):

        self.burnin = 0
        super(DECOLLEBase, self).__init__()

        self.LIF_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()

    def __len__(self):
        return len(self.LIF_layers)

    def step(self, data_batch):
        raise NotImplemented('')

    def forward(self, data_batch, doinit=True, return_sequence=False, *args, **kwargs):
        '''
        Run network on *data_batch* sequence.
        *args*
        data_batch : Sequence has shape [batch_size, time]+[input_shape]
        doinit : Do an state init prior to running
        return_sequence : Return output of all layers and states
        '''
        if doinit: self.init(data_batch)
        t_sample = data_batch.shape[1]
        out = []
        for t in (range(self.burnin,t_sample)):
            data_batch_t = data_batch[:,t]
            out_ = self.step(data_batch_t, *args, **kwargs)
            if return_sequence: out.append(out_)
        if not return_sequence:
            out = out_[-1][-1]
            
        return out 

    @property
    def output_layer(self):
        return self.readout_layers[-1]

    def name_param(self):
        return self.named_parameters()

    def get_trainable_parameters(self, layer=None):
        if layer is None:
            params = chain(*[l.parameters() for l in self.LIF_layers])
        else:
            params = self.LIF_layers[layer].parameters()
        params = (param for param in params if param.requires_grad)
        return params

    def get_trainable_named_parameters(self, layer=None):
        if layer is None:
            params = dict()
            for k,p in self.named_parameters():
                if p.requires_grad:
                    params[k]=p

            return params
        else:
            return self.LIF_layers[layer].named_parameters()

    def get_trainable_named_parameters(self, layer=None):
        if layer is None:
            params = dict()
            for k,p in self.named_parameters():
                if p.requires_grad:
                    params[k]=p

            return params
        else:
            return self.LIF_layers[layer].named_parameters()

    def init(self, data_batch, burnin = None):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        if burnin is None:
            burnin = self.burnin
        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None
        with torch.no_grad():
            for t in (range(0,max(self.burnin,1))):
                data_batch_t = data_batch[:,t]
                out_ = self.step(data_batch_t)

        for l in self.LIF_layers: state_detach(l.state)

    def init_parameters(self, data_batch):
        with torch.no_grad():
            Sin_t = data_batch[:, 0, :, :]
            s_out, r_out = self.step(Sin_t)[:2]
            ins = [self.LIF_layers[0].state.Q]+s_out
            for i,l in enumerate(self.LIF_layers):
                l.init_parameters(ins[i])

    def reset_lc_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        self.reset_lc_bias_parameters(layer,lc_ampl)

    def reset_lc_bias_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)
    
    def get_input_layer_device(self):
        if hasattr(self.LIF_layers[0], 'get_device'):
            return self.LIF_layers[0].get_device() 
        else:
            return list(self.LIF_layers[0].parameters())[0].device

    def get_output_layer_device(self):
        return self.output_layer.weight.device 

    def process_output(net, data_batch):
        '''
        Process the outputs of step run over entire sequence data_batch as a continugous array.
        *data_batch*: batch of inputs, same shape as for data_batch in step()
        '''
        with torch.no_grad():
            from decolle.utils import tonp
            net.init(data_batch)
            t = (data_batch.shape[1],)
            out_states = net.step(data_batch[:,0])
            readouts = [None for _ in net.output_statenames]
            for k,v in net.output_statenames.items():
                readouts[v] = [np.zeros(t+tonp(layer).shape     ) for layer in out_states[v] if layer is not None]

            for t in range(data_batch.shape[1]):
                net.state = None
                out_states = net.step(data_batch[:,t])
                for i in range(len(net.LIF_layers)):
                    for k,v in net.output_statenames.items():
                        if out_states[v] is not None:
                            if out_states[v][i] is not None:
                                readouts[v][i][t,:] = [tonp(output) for output in out_states[v][i]]

        return readouts


class DECOLLELoss(object):
    def __init__(self, loss_fn, net, reg_l = None):
        self.loss_fn = loss_fn
        self.nlayers = len(net)
        self.num_losses = len([l for l in loss_fn if l is not None])
        if len(loss_fn)!=self.nlayers:
            warnings.warn("Mismatch is in number of loss functions and layers. You need to specify one loss function per layer")
        self.reg_l = reg_l
        if self.reg_l is None: 
            self.reg_l = [0 for _ in range(self.nlayers)]

    def __len__(self):
        return self.nlayers

    def __call__(self, s, r, u, target, mask=1, sum_=True):
        loss_tv = []
        for i,loss_layer in enumerate(self.loss_fn):
            if loss_layer is not None:
                loss_tv.append(loss_layer(r[i]*mask, target*mask))
                if self.reg_l[i]>0:
                    uflat = u[i].reshape(u[i].shape[0],-1)
                    reg1_loss = self.reg_l[i]*1e-2*((relu(uflat+.01)*mask)).mean()
                    reg2_loss = self.reg_l[i]*6e-5*relu((mask*(.1-sigmoid(uflat))).mean())
                    loss_tv[-1] += reg1_loss + reg2_loss

        if sum_:
            return sum(loss_tv)
        else:
            return loss_tv


class StatelessWrapper(torch.nn.Module):
    '''Wrapper module to extract the state variables from the LIF layers and loop them as input and output'''

    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net

    def step(self, input, external_states):
        # Looping through LIF layers and replace the internal_states with the external_states
        for lif_layer, layer_external_states in zip(self.base_net.LIF_layers, external_states):
            if lif_layer.state is None:
                lif_layer.state = lif_layer.NeuronState(*layer_external_states)
            else:
                for name, external_state in zip(lif_layer.state._fields, layer_external_states):
                    assert getattr(lif_layer.state, name).shape == external_state.shape
                    lif_layer.state = lif_layer.state._replace(**{name: external_state})
        outputs = self.base_net.step(input)
        external_states = []
        for lif_layer in self.base_net.LIF_layers:
            layer_states = {}
            for name, internal_state in lif_layer.state._asdict().items():
                layer_states[name] = internal_state.detach().clone()
            external_states.append(lif_layer.NeuronState(**layer_states))
        return outputs, external_states

    # A forward function is still required for onnx export.
    forward = step

    def __len__(self):
        return len(self.base_net)
