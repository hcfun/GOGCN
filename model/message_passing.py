import inspect, torch
from torch_scatter import scatter

def scatter_(name, src, index, dim_size=None):
    """

    :param name: The aggregation to use (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
    :param src: The source tensor
    :param index: The indices of elements to scatter.
    :param dim_size: Automatically create output tensor with size
			        :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			        minimal sized output tensor is returned. (default: :obj:`None`)
    :return: The result tensor after scatter according to index
    """
    if name == 'add': name = 'sum'
    assert name in ['sum', 'mean', 'max']

    out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
    return out[0] if isinstance(out, tuple) else out


    #a sample using scatter function
    '''
	a=torch.tensor([[1,2,3,4,5],
					[1,3,5,7,9],
					[2,4,6,8,10]])
	b=torch.tensor([0,1,0])
	c=scatter(a,b,dim=0,out=None,dim_size=2,reduce=name)
	print(c)
	'''




class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
    	\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
    	\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
    	\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getargspec(self.message)[0][1:] #get the args list of message function(Start with the second parameter) ['x_j', 'edge_type', 'rel_embed', 'edge_norm', 'mode']
        self.update_args = inspect.getargspec(self.update)[0][2:] #get the args list of message function(Start with the third parameter) []
        self.aggr = aggr
        assert aggr in ['add', 'mean', 'max']

    def propagate(self, aggr, edge_index, **kwargs):
        """

        :param aggr:
        :param edge_index: adjacency matrix
        :param kwargs: args dict which contains all args.   kwargs.keys(): ['x', 'edge_type', 'rel_embed', 'edge_norm', 'mode', 'edge_index']
        :return:
        """

        kwargs['edge_index'] = edge_index
        size = None
        message_args =[]

        #self.message_args: ['x_j', 'edge_type', 'rel_embed', 'edge_norm', 'mode']
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]] #ent_embedding
                size = tmp.size(0) #num_ent
                message_args.append(tmp[edge_index[0]]) #sub_ent_embedding. Lookup for head entities in edges
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]] #ent_embedding==kwags['x']
                size = tmp.size(0) #num_ent
                message_args.append(tmp[edge_index[1]]) #obj_ent_embedding. Lookup for tail entities in edges. shape:(num_edge, 100)
            else:
                message_args.append(kwargs[arg]) #Take things from kwargs

        update_args = [kwargs[arg] for arg in self.update_args]  #Take update args from kwargs
        out = self.message(*message_args) #shape:(num_edge, 200)
        out = scatter_(aggr, out, edge_index[0], dim_size=size) #use 'sum' style to updated ent_embedding
        out = self.update(out, *update_args)
        return out


    def message(self, x_j):

        return x_j

    def update(self, aggr_out):

        return aggr_out


