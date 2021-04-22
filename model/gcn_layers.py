from model.message_passing import MessagePassing
from helper import *


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
        """

        :param in_channels: The initial embed_dim
        :param out_channels: The updated embed_dim
        :param num_rels: The number of relation
        :param act:
        :param params:
        """
        super(self.__class__, self).__init__()


        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device =None

        #let every neighbor point to sub entity (if sub ent is head, using Wo; if sub ent is tail, using Wi)
        self.w_loop = get_param((in_channels, out_channels)) #Ws
        self.w_in = get_param((in_channels, out_channels)) #Wi
        self.w_out = get_param((in_channels, out_channels)) #Wo

        self.w_rel = get_param((in_channels, out_channels)) #Wrel
        self.loop_rel = get_param((1, in_channels)) #may be not use

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.W = get_param((out_channels, out_channels))#The new weight of the gcnlayer

        if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))# add a parameter to the model named bias


    def forward(self, x, edge_index, edge_type, rel_embed):
        """

        :param x: ent_embedding.shape(num_ent, 100)
        :param edge_index: adjacency matrix. shape(2,num_edge)
        :param edge_type: shape(num_edge)
        :param rel_embed: rel_embedding. shape(num_rel*2, 100) take the reverse relations into account
        :return:
        """
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0) #add self loop into rel_embedding. shape:(num_rel*2+1, 100)
        num_edges = edge_index.size(1) // 2 #delete reverse edge. real num_edge
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:] #shape(2: num_edges). separate the real edges and reverse edges.
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:] #shape(num_edges)

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device) #shape(2,num_ent) row1=row2:[0,1,2,...,num_ent-1]
        self.loop_type = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device) #shape(num_ent) [num_rel*2,num_rel*2,...,num_rel*2]

        self.in_norm = self.compute_norm(self.in_index, num_ent) #shape:(num_edge)
        self.out_norm = self.compute_norm(self.out_index, num_ent) #shape:(num_edge)

        #update ent_embedding
        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_norm=self.in_norm, mode='in')  #use tail to update head
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop') #use self-loop to update itself\
        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        out = torch.mm(out, self.W)

        if self.p.bias: out = out + self.bias
        out = self.bn(out)

        #torch.matmul(rel_embed, self.w_rel)[:-1]: update rel_embeding and [:-1] refer to delete the last row, that is, ignoring the self loop inserted
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted



    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        """

        :param x_j: obj_ent_embedding. shape:(num_edge, 100)
        :param edge_type: edge corresponding to the edge_index
        :param rel_embed: rel_embedding
        :param edge_norm:
        :param mode: in out loop rel
        :return:
        """
        weight = getattr(self, 'w_{}'.format(mode)) #Get the weight defined above according to the mode.
        rel_embed = torch.index_select(rel_embed, 0, edge_type) #Get the rel_embedding corresponding to current edge according to the edge_type
        xj_rel = self.rel_transform(x_j, rel_embed) #shape:(num_edge, 100)
        out = torch.mm(xj_rel, weight) #shape:(num_edge, 200)
        return out if edge_norm is None else out*edge_norm.view(-1, 1) #shape:(num_edge, 200)





    def update(self, aggr_out):

        return aggr_out

    def rel_transform(self, ent_embed, rel_embed):
        """
        message. corresponding the composition operator: φ(es,er)

        :param ent_embed: sub_ent_embedding
        :param rel_embed: rel_embeding
        :return: The tensor after composition operating
        """

        if self.p.opn == 'corr': trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub': trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult': trans_embed = ent_embed * rel_embed
        else: raise NotImplementedError

        return trans_embed

    def compute_norm(self, edge_index, num_ent):
        """
        D^{-1} * (A+I) or D^{-0.5} * (A+I) *  D^{-0.5}

        :param edge_index: adjacency matrix
        :param num_ent:
        :return:
        """

        row, col = edge_index
        edge_weight = torch.ones_like(row).float()

        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent) #calcuate the out degree of an node
        deg_inv = deg.pow(-0.5) #D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm


    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)







