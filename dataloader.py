from helper import *
from torch.utils.data import Dataset



class Dataset(Dataset):
    """
    Train Dataset class.

    Parameters
    ----------------------------------------------------
    triples:  The triples used for train the model
    params:   Parameters for the experiments

    Returns
    ----------------------------------------------------
    A training Dataset class instance used by Dataloader
    """
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32) # [0,1,…,self.p.num_ent-1]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx] #get the triple according to the idx
        triple, label, sub_samp = torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
        trp_label = self.get_label(label)

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0/self.p.num_ent)

        return triple, trp_label, None, None


    def get_label(self,label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for ent in label: y[ent]=1.0
        return torch.FloatTensor(y)

    @staticmethod
    #merge the 128 (triple,label) to a batch.
    #data:from the __getitem__ function
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)#shape:(batch, num_ent) value=1 when triple is true else 0.Thait is, it's convenient to sample neg entities
        return triple, trp_label

    #negative sampling. Don't need.
    def get_neg_ent(self, triple, label):
        def get(triple, label):
            pos_obj = label
            mask = np.ones([self.p.num_ent], dtype=bool)
            mask[label] =0
            neg_ent = np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape[-1]
            neg_ent = np.concatenate((pos_obj.reshape[-1]), neg_ent)

            return neg_ent
        neg_ent = get(triple, label)
        return neg_ent