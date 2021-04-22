from dataloader import *
from model.model import *
import numpy as np
import pandas as pd




class Main(object):
    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        -----------------------------------------------------
        params:         List of hyper-parameters of the model

        Returns
        -----------------------------------------------------
        Creates computational graph and optimizer

        """
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        self.logger.info(vars(self.p))
        pprint(vars(self.p))
        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model()
        self.optimizer = self.add_optimizer(self.model.parameters())
        self.last_embedding = torch.rand([self.p.num_ent, self.p.embed_dim])




    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        --------------------------------------------------------------------

        Returns
        --------------------------------------------------------------------
        self.ent2id:            Entity to unique identifier mapping
        self.ent2id:            Inverse mapping of self.ent2id
		self.id2rel:            Inverse mapping of self.rel2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.data_iter:		    The dataloader for different data splits
        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for line in open ('./data/triples'):
            sub, rel, obj = map(str.lower, line.strip().split('\t')) #str.lower: Convert uppercase letters to lowercase letters. the deatils of function 'map' can be seen at:https://www.cnblogs.com/xaiobong/p/9955902.html
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)}) #add reverse relation to rel2id

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        print("num of entities:{}".format(self.p.num_ent).title())
        self.p.num_rel = len(self.rel2id) // 2
        print("num of relations:{}".format(self.p.num_rel).title())
        self.p.embed_dim = self.p.k_w *self.p.k_h if self.p.embed_dim is None else self.p.embed_dim #for matching conve's config


        self.data = ddict(list) #store all data
        sr2o = ddict(set) #(sub,rel) to obj mapping

        #construct the dataset
        for line in open('./data/triples'):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
            self.data['triples'].append((sub, rel, obj))

            sr2o[(sub, rel)].add(obj)
            sr2o[(obj, rel+self.p.num_rel)].add(sub)


        self.data = dict(self.data)
        self.sr2o = {k:list(v) for k, v in sr2o.items()} #(sub,rel) to obj mapping in 'train'

        self.sr2o_All = {k:list(v) for k, v in sr2o.items()} #(sub,rel) to obj mapping in all data (for filtered setting)
        self.triples = ddict(list)

        #construct triples['train'], triples['test_tail'], triples['test_head'], triples['valid_tail'], triples['valid_head']
        for (sub, rel), obj in self.sr2o.items():
            self.triples['triples'].append({'triple':(sub, rel, -1), 'label':self.sr2o[(sub,rel)], 'sub_samp':1}) #the label of (sub,rel) is obj and the number of neg sample for a pos triple is all entity.

        self.triples = dict(self.triples)

        def get_dataloader(dataset_class, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples['triples'], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )


        self.data_iter = {
            'train':        get_dataloader(Dataset, self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.bulid_adj()


    def bulid_adj(self):
        """
        Bulid the adjacency matrix of KG

        Parameters
        ------------------------------------

        Rerurn
        ------------------------------------
        The adjacency matrix of KG for GCN
        """
        edge_index, edge_type = [], []
        for sub, rel, obj in self.data['triples']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        #add the inverse edges
        for sub, rel, obj in self.data['triples']:
            edge_index.append((obj, sub))
            edge_type.append(rel+self.p.num_rel)

        #convert to tensor and deploy into gpu
        edge_index = torch.LongTensor(edge_index).to(self.device).t() #shape:(2, len(data['train'])) row1:sub, row2:obj
        edge_type = torch.LongTensor(edge_type).to(self.device) #shape:(len(data['train']))

        return edge_index, edge_type


    def read_batch(self, batch):
        """
        Read a batch of data and move the tensors in batch to device

        Parameters
        -------------------------------
        batch: the data batch to process

        Return
        -------------------------------
        heads, rel, tail, label

        """
        triple, label = [_.to(self.device) for _ in batch]
        return triple[:, 0], triple[:, 1], triple[:, 2], label


    def add_model(self):
        """
        Create the copmutional graph

        Return
        ---------------------
        The model to be used
        """

        model = GCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        model.to(self.device)
        return model



    def add_optimizer(self, parameters):
        """
        Create an optimizer for training process

        :param parameters: The parameters of the model

        :return: an optimizer for learning the parameters of the model
        """

        return torch.optim.Adam(parameters, lr=self.p.lr,  weight_decay=self.p.l2)



    def save_model(self, save_path):
        """
        Function to save a model. It savesthe model parameters, best validation score,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run

        :save_path: path to save the model
        :return:
        """

        state = {
            'state_dict': self.model.state_dict(),
            'min_loss': self.min_loss,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)


    def load_model(self, load_path):
        """
        Function to load a saved model


        :load_path: path to the saved model:
        """

        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.min_loss = state['min_loss']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])


    def run_epoch(self, epoch):
       """
       Function to run one epoch of training

       :param epoch:current epoch count
       :param val_mrr:
       :return: loss:the loss value after current epoch
       """
       self.model.train()
       losses=[]
       train_iter = iter(self.data_iter['train'])

       for step, batch in enumerate(train_iter):
           self.optimizer.zero_grad()
           sub, rel, obj, label = self.read_batch(batch)
           pred, all_ent_embedding = self.model.forward(sub, rel)
           loss = self.model.loss(pred, label)

           loss.backward()
           self.optimizer.step()
           losses.append(loss.item())

           if step % 100 == 0:
               self.logger.info('[E:{}| {}]: Train Loss:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.p.name))

       print('-'*25)
       loss = np.mean(losses)
       return loss, all_ent_embedding

    def train(self):
        """
        Function to run training and evaluation of model

        """
        self.min_loss = 100
        save_path = os.path.join('./checkpoints', self.p.name) #The path to save the model
        embed_path = os.path.join(self.p.embed_dir, 'ent_embedding_{}'.format(self.p.name))

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss, all_ent_embedding = self.run_epoch(epoch)

            if self.min_loss > train_loss:
                self.min_loss = train_loss
                self.best_epoch = epoch
                self.save_model(save_path)
                self.last_embedding = all_ent_embedding
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 ==0  and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger .info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 25:
                    self.logger.info("Early Stopping!!")
                    break
        emedding = self.last_embedding.clone().detach().float().cpu().numpy()
        pd.DataFrame(emedding).to_csv('{}.csv'.format(embed_path), index=None, header=None)








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='Pretraining_' + str(uuid.uuid4())[:8], help='Set run name for saving/restoring models')
    parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in model')
    parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim', dest='init_dim', default=100, type=int, help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim', dest='embed_dim', default=None, type=int, help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int, help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
    parser.add_argument('-embed_dir', dest='embed_dir', default='./embedding/', help='Config directory')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())

    model = Main(args)
    model.train()




