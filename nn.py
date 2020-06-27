from utils import *
from data import *
import torch.nn as nn
import datetime
import math
import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='trivago_imp', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample/trivago_imp')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=128, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.all_cat_columns = self.config.all_cat_columns  # 所有的categorical columns
        self.categorical_emb_dim = config.categorical_emb_dim
        self.hidden_dims = config.hidden_dims
        self.num_embeddings = config.num_embeddings

        # embedding part
        self.emb_dict = torch.nn.ModuleDict()
        for cat_col in self.config.all_cat_columns:
            if cat_col =='item_id':
                
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim, padding_idx = self.config.transformed_dummy_item)
            else:
                # 所有categorical columns各自做embedding
                self.emb_dict[cat_col] = torch.nn.Embedding(num_embeddings=self.num_embeddings[cat_col],
                                                            embedding_dim=self.categorical_emb_dim)
        # gru for extracting session and user interest
        self.gru_sess = torch.nn.GRU(input_size = self.categorical_emb_dim *2, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=2, batch_first=True)
        self.other_item_gru = torch.nn.GRU(input_size = self.categorical_emb_dim, hidden_size = self.categorical_emb_dim//2, bidirectional=True , num_layers=1, batch_first=True)
        
        # linear layer on top of continuous features
        self.cont_linear = torch.nn.Linear(config.continuous_size,self.categorical_emb_dim )

        # hidden layerrs
        self.hidden1 = torch.nn.Linear(self.categorical_emb_dim*17 , self.hidden_dims[0])
        self.hidden2 = torch.nn.Linear(self.hidden_dims[0] + config.continuous_size*2 + 3 + config.neighbor_size, self.hidden_dims[1] )
            
        # output layer
        self.output = torch.nn.Linear(self.hidden_dims[1], 1)
        
        # batch normalization
        self.bn = torch.nn.BatchNorm1d(self.categorical_emb_dim*17)
        self.bn_hidden = torch.nn.BatchNorm1d(self.hidden_dims[0] + config.continuous_size*2+ 3 + config.neighbor_size )

        self.srgnn = SessionGraph(opt, 446006)
        
    def forward(self, item_id, past_interactions, mask, price_rank, city, last_item, impression_index, cont_features, star, past_interactions_sess, past_actions_sess, last_click_item, last_click_impression, last_interact_index, neighbor_prices, other_item_ids, city_platform, batch_size, slices, data):
        embeddings = []
        user_embeddings = []
        batch_size = item_id.size(0)

        # todo 讓 SR-GNN也共用 item embedding
        # embedding of all categorical features
        emb_item = self.emb_dict['item_id'](item_id)
        emb_past_interactions = self.emb_dict['item_id'](past_interactions)
        emb_price_rank = self.emb_dict['price_rank'](price_rank)
        emb_city = self.emb_dict['city'](city)
        emb_last_item = self.emb_dict['item_id'](last_item)
        emb_impression_index = self.emb_dict['impression_index'](impression_index)
        emb_star = self.emb_dict['star'](star)
        emb_past_interactions_sess = self.emb_dict['item_id'](past_interactions_sess)  # reference/ click_out item
        emb_past_actions_sess = self.emb_dict['action'](past_actions_sess)
        emb_last_click_item = self.emb_dict['item_id'](last_click_item)
        emb_last_click_impression = self.emb_dict['impression_index'](last_click_impression)
        emb_last_interact_index = self.emb_dict['impression_index'](last_interact_index)
        emb_city_platform = self.emb_dict['city_platform'](city_platform)
        emb_other_item_ids = self.emb_dict['item_id'](other_item_ids)
        
        # other items processed by gru
        emb_other_item_ids_gru, _ = self.other_item_gru(emb_other_item_ids)
        pooled_other_item_ids = F.max_pool1d(emb_other_item_ids_gru.permute(0,2,1), kernel_size=emb_other_item_ids_gru.size(1)).squeeze(2)

        # user's past clicked-out item
        # emb_past_interactions = emb_past_interactions.permute(0,2,1)  # (1024, 10, 128) to (1024, 128, 10)
        # pooled_interaction = F.max_pool1d(emb_past_interactions, kernel_size=self.config.sequence_length).squeeze(2)  # (1024, 128)

        # fixme 改成用 index取 而非 slice
        # for i, j in zip(slices, np.arange(len(slices))):
            # model.optimizer.zero_grad()
            # 這裡替換成 SR-GNN的 session emb
            # _, pooled_interaction = forward(self.srgnn, i, data)

        _, pooled_interaction = forward(self.srgnn, slices, data, self.emb_dict['item_id'])

        # concatenate sequence of item ids and actions to model session dynamics
        emb_past_interactions_sess = torch.cat([emb_past_interactions_sess, emb_past_actions_sess], dim=2)
        emb_past_interactions_sess, _ = self.gru_sess(emb_past_interactions_sess)
        emb_past_interactions_sess = emb_past_interactions_sess.permute(0, 2, 1)
        pooled_interaction_sess = F.max_pool1d(emb_past_interactions_sess, kernel_size=self.config.sess_length).squeeze(2)
        
        
        # categorical feature interactions (做element wise的相乘)
        item_interaction = emb_item * pooled_interaction
        item_last_item = emb_item * emb_last_item
        item_last_click_item = emb_item * emb_last_click_item
        # item_interaction = torch.mm(emb_item, pooled_interaction.T)
        # item_last_item = torch.mm(emb_item, emb_last_item.T)
        # item_last_click_item = torch.mm(emb_item, emb_last_click_item.T)
        imp_last_idx = emb_impression_index * emb_last_interact_index

        # efficiently compute the aggregation of feature interactions 
        emb_list = [emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, emb_star]
        emb_concat = torch.cat(emb_list, dim=1)
        sum_squared = torch.pow( torch.sum( emb_concat, dim=1), 2).unsqueeze(1)
        squared_sum = torch.sum( torch.pow( emb_concat, 2), dim=1).unsqueeze(1)
        second_order = 0.5 * (sum_squared - squared_sum)
        
        # compute the square of continuous features
        squared_cont = torch.pow(cont_features, 2)


        # DNN part
        concat = torch.cat([emb_item, pooled_interaction, emb_price_rank, emb_city, emb_last_item, emb_impression_index, item_interaction, item_last_item, emb_star, pooled_interaction_sess, emb_last_click_item, emb_last_click_impression, emb_last_interact_index, item_last_click_item, imp_last_idx, pooled_other_item_ids, emb_city_platform], dim=1)
        concat = self.bn(concat)
        
        hidden = torch.nn.ReLU()(self.hidden1(concat))

        hidden = torch.cat([cont_features, hidden, sum_squared, squared_sum, second_order, squared_cont, neighbor_prices], dim=1)
        
        hidden = self.bn_hidden(hidden)
        hidden = torch.nn.ReLU()(self.hidden2(hidden))

        output = torch.sigmoid(self.output(hidden)).squeeze()

        return output


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)  # item embedding
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        self.linear_concat = nn.Linear(self.hidden_size*26, 25, bias=True)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # session emb
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        # 改成回傳 session emb
        return a

    def forward(self, inputs, A):
        # hidden = self.embedding(inputs)
        hidden = inputs
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, item_emb):
    alias_inputs, A, items, mask, targets, imp = data.get_slice(i)

    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    # alias_inputs = trans_to_cpu(torch.Tensor(alias_inputs).long())
    # items = trans_to_cpu(torch.Tensor(items).long())
    # A = trans_to_cpu(torch.Tensor(A).float())
    # mask = trans_to_cpu(torch.Tensor(mask).long())

    # todo 這裡的 items先過 embedding
    items = item_emb(items)
    hidden = model(items, A)  # run model.forward()
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)

