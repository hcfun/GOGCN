# GOGCN

The resources of Graph convolutional network on Gene Ontology for measuring gene functional similarity


## Dependencies
- Install [PyTorch](https://pytorch.org/) using [Anaconda](https://www.anaconda.com/products/individual).
- Install all the requirements from `requirements.txt`. You can download the torch_scatter corresponding to your cuda and python version at https://pytorch-geometric.com/whl/torch-1.6.0.html and then use 'pip install ...' command to install it.


## Datasets
- [Gene Ontology](http://geneontology.org/docs/download-ontology/) dated September 2020.
- [Gene Ontology annotations](http://geneontology.org/docs/download-go-annotations/) for Homo sapiens(dated October 2020) and Saccharomy cescerevisiae(dated October 2020).

## Train model and compute gene functional similarity
- run `run.py` to train the model and learn the representation for terms and relations.
- run `geneSim.py` to compute gene functional similarity.

## Note
- There are some absolute paths in `Comparison_algorithm/SORA/SORA.py` and `Comparison_algorithm/Wang/SV.py`. Please change these absolue paths when reproducing the results of GOGCN.
- run `preprocess.sh` first.
