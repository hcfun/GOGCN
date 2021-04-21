# GOGCN

The resources of Graph convolutional network on Gene Ontology for measuring gene functional similarity


## Dependencies
- Install [PyTorch](https://pytorch.org/) using [Anaconda](https://www.anaconda.com/products/individual).
- Install all the requirements from `requirements.txt`. You can download the torch_scatter corresponding to your cuda and python version at (https://pytorch-geometric.com/whl/torch-1.6.0.html).


## Datasets
- [Gene Ontology](http://geneontology.org/docs/download-ontology/) dated September 2020.
- [Gene Ontology annotations](http://geneontology.org/docs/download-go-annotations/) for Homo sapiens(dated October 2020) and Saccharomy cescerevisiae(dated October 2020).

## Train model and compute gene functional similarity
- run `run.py` to train the model. 
- run `geneSim.py` to compute gene functional similarity.
