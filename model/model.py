from helper import *
from model.gcn_layers import GCNConv

class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()


    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class GCNBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        """

        :param edge_index: adjacency matrix which contains reverse relation. shape:(2, num_edge*2) row1:sub_ent; row2: obj_ent
        :param edge_type: shape:(num_edge*2)
        :param num_rel: num of rel(Do not contain reverse relation)
        :param params: p
        """
        super(GCNBase, self).__init__(params)
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.ent_embedding = get_param((self.p.num_ent, self.p.init_dim)) #initialize the ent_embedding shape:(num_ent, 100)
        self.device = self.edge_index.device

        self.rel_embedding = get_param((num_rel*2, self.p.init_dim))

        self.conv1 = GCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)

        self.conv2 = GCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None


        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    def forward_base(self, sub, rel, drop1, drop2):
        """
        :param sub: sub_ent_index
        :param rel: rel_ent_index
        :return: sub_emb:updated sub_ent_embedding. rel_emb:updated rel_embedding. x:updated all ent_embedding
        """
        r = self.rel_embedding if self.p.score_func != 'transe' else torch.cat([self.rel_embedding, -self.rel_embedding], dim=0)
        x, r = self.conv1(self.ent_embedding, self.edge_index, self.edge_type, rel_embed=r)
        x = drop1(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x, r)
        x = drop2(x) if self.p.gcn_layer == 2 else x #shape:(num_ent, 200)

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x


class GCN_ConvE(GCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2*self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim) #shape:(128,1,200)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim) #shape:(128,1,200)
        stack_inp = torch.cat([e1_embed, rel_embed], 1) #shape:(128,2,200)
        stack_inp = torch.transpose(stack_inp, 2 ,1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h)) #shape:(128,1,20,20)
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent_emb = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb) #shape:(128,1,20,20)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x) #shape:(128,200,14,14)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz) #shape:(128, 200*14*14)
        x = self.fc(x) #shape:(128,200)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x) #shape:(128,200)

        x = torch.mm(x, all_ent_emb.transpose(1,0)) #shape:(128,14541)
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score, all_ent_emb