# Deep Continuous Local Learning (DECOLLE) with Hessian Aware Quantization

DECOLLE is an online learning framework for spiking neural networks.
The algorithmic details are described in this [Frontiers paper](https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full).
If you use this work in your research, please cite as:

```
@ARTICLE{decolle2020,
AUTHOR={Kaiser, Jacques and Mostafa, Hesham and Neftci, Emre},
TITLE={Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)},
JOURNAL={Frontiers in Neuroscience},
VOLUME={14},
PAGES={424},
YEAR={2020},
URL={https://www.frontiersin.org/article/10.3389/fnins.2020.00424},
DOI={10.3389/fnins.2020.00424},
ISSN={1662-453X}
```

This repo includes quantization of the network with QPytorch, with bit-precision guided by the layer-wise Hessian trace, as well as a simpliefied neuron model `SimpleLIFLayer`.
More details [here](https://arxiv.org/abs/2104.14117).
If you use this work in your research, please cite as:

```
@misc{lui2021hessian,
      title={Hessian Aware Quantization of Spiking Neural Networks}, 
      author={Hin Wai Lui and Emre Neftci},
      year={2021},
      eprint={2104.14117},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

### Installing
Clone and install. The Python setuptools will take care of dependencies
```
https://github.com/luithw/decolle-quantization.git
cd decolle-quantization
python setup.py install --user
```

The following will run decolle on the default parameter set with full precision. 
This scripts will also compute and printout the layer-wise Hessian trace.
```
cd scripts
python train_lenet_decolle.py
```

To enable quantization, add the quantization flag.
```
python train_lenet_decolle.py --quantization
```

You can change the bit-precision of each layer by modifying `scripts/parameters/params_nmnist_simplelif_full.yml`.

## Authors


* **Emre Neftci** - *Initial work* - [eneftci](https://github.com/eneftci)
* **Jacques Kaiser** - [jackokaiser](https://github.com/jackokaiser)
* **Massi Iacono** - [miacono](https://github.com/miacono)
* **Hin Wai Lui** - *Hessian Aware Quantization* - [hwlui](https://github.com/luithw)


## License

This project is licensed under the GPLv3 License - see the [LICENSE.txt](LICENSE.txt) file for details
