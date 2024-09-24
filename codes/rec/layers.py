import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from .activate import activation_layer

class DNN(nn.Module):
    """ MLP的全连接层
    """
    def __init__(self, 
                 input_dim,
                 hidden_units,
                 activation='relu',
                 dropout_rate=0,
                 use_bn=False,
                 init_std=1e-4,
                 dice_dim=3
                 ) -> None:
        super(DNN, self).__init__()

        assert isinstance(hidden_units, (tuple, list)) and len(hidden_units) > 0, 'hidden_unit support non_empty list/tuple inputs'
        self.dropout = nn.Dropout(dropout_rate)
        hidden_units = [input_dim] + list(hidden_units)

        layers = []

        for i in range(len(hidden_units) - 1):
            # Linear
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            # BN
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_units[i + 1]))
            # Activation
            layers.append(activation_layer(activation, hidden_units[i + 1], dice_dim))
            # Dropout
            layers.append(self.dropout)

        self.layers = nn.Sequential(*layers)

        for name, tensor in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        # inputs: [btz, ..., input_dim]
        return self.layers(inputs)  # [btz, ..., hidden_units[-1]]

class ResidualNetwork(nn.Module):
    """ 残差链接
    """
    def __init__(self,
                 input_dim,
                 hidden_units,
                 activation="relu",
                 dropout_rate=0,
                 use_bn=False,
                 init_std=1e-4,
                 dice_dim=3
                 ) -> None:
        super(ResidualNetwork, self).__init__()

        assert isinstance(hidden_units, (tuple, list)) and len(hidden_units) > 0, 'hidden_unit support non_empty list/tuple inputs'
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        layers = []
        for i in range(len(hidden_units)):
            # Linear
            layers.append(nn.Linear(input_dim, hidden_units[i]))
            # BN
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_units[i]))
            # Activate
            layers.append(activation_layer(activation, hidden_units[i], dice_dim))
            # Linear
            layers.append(self.dropout)
            
            self.layers.append(nn.Sequential(*layers))
        
        for layer in self.layers:
            for name, tensor in layer.named_parameters():
                if "weight" in name:
                    nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        # inputs: [btz, ..., input_dim]
        raw_inputs = inputs
        for layer in self.layers:
            inputs = raw_inputs + layer(inputs)
        # [btz, ..., input_dim]
        return inputs

class PredictionLayer(nn.Nodule):
    def __init__(self,
                 out_dim=1,
                 use_bais=True,
                 logit_transform=None,
                 **kwargs
                 ) -> None:
        super(PredictionLayer, self).__init__()
        self.logit_transform = logit_transform
        if use_bais:
            self.bias = nn.Parameter(torch.zeros((out_dim,)))
    
    def forward(self, x):
        output = x
        if hasattr(self, "bias"):
            output += self.bias
        
        if self.logit_transform == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.logit_transform == "softmax":
            output = torch.softmax(output, dim=-1)
        return output
    
class FM(nn.Module):
    """ FM因子分解的实现，使用二阶项简化来计算交叉部分
    inputs: [btz, field_size, emb_size]
    output: [btz, 1]
    """
    def __init__(self, ) -> None:
        super(FM, self).__init__()
    
    def forward(self, inputs):
        # inputs: [btz, field_size, emb_size]
        square_sum = torch.pow(torch.sum(inputs, dim=1, keepdim=True), 2)   # [btz, 1, emb_size]
        sum_square = torch.sum(torch.pow(inputs, 2), dim=1, keepdim=True)   # [btz, 1, emb_size]

        return 0.5 * torch.sum(square_sum - sum_square, dim=-1)


class SequencePoolingLayer(nn.Module):
    """ Seq 输入转pooling，支持多种pooling方式
    """
    def __init__(self, mode="mean", support_masking=False) -> None:
        super(SequencePoolingLayer, self).__init__()
        assert mode in {"sum", "mean", "max"}, "parameter mode should in [sum, mean, max]"
        self.mode = mode
        self.support_masking = support_masking
    
    def forward(self, seq_value_len_list):
        # seq_value_len_list: [btz, seq_len, hdsz], [btz, seq_len]/[btz,1]
        seq_input, seq_len = seq_value_len_list

        if self.support_masking:    # 传入的是mask
            mask = seq_len.float()
            user_behavior_len = torch.sum(mask, dim=-1, keepdim=True)   # [btz, 1]
            mask = mask.unsqueeze(2)
        else:   # 传入的是behavior长度
            user_behavior_len = seq_len
            mask = torch.arange(0, seq_input.shape[1]) < user_behavior_len.unsqueeze(-1)
            mask = torch.transpose(mask, 1, 2)  # [btz, seq_len, 1]

        mask = torch.repeat_interleave(mask, seq_input.shape[-1], dim=2)  # [btz, seq_len, hdsz]
        mask = (1 - mask).bool()

        if self.mode == 'max':
            seq_input = torch.masked_fill(seq_input, mask, 1e-8)
            return torch.max(seq_input, dim=1, keepdim=True)  # [btz, 1, hdsz]
        elif self.mode == 'sum':
            seq_input = torch.masked_fill(seq_input, mask, 0)
            return torch.sum(seq_input, dim=1, keepdim=True)  # [btz, 1, hdsz]
        elif self.mode == 'mean':
            seq_input = torch.masked_fill(seq_input, mask, 0)
            seq_sum = torch.sum(seq_input, dim=1, keepdim=True)
            return seq_sum / (user_behavior_len.unsqueeze(-1) + 1e-8)

class AttentionSequencePoolingLayer(nn.Module):
    """ DIN 中使用的序列注意力
    """
    def __init__(self,
                 att_hidden_units=(80, 40),
                 att_activation='sigmoid',
                 weight_normalization=False,
                 return_score=False,
                 embedding_dim=4,
                 **kwargs
                 ) -> None:
        super(AttentionSequencePoolingLayer, self).__init__()
        self.return_score = return_score
        self.weight_normalization = weight_normalization
        # 局部注意力
        self.dnn = DNN(
            input_dim=4 * embedding_dim,
            hidden_units=att_hidden_units,
            activation=att_activation,
            dice_dim=kwargs.get("dice_dim", 3),
            use_bn=kwargs("use_bn", False),
            dropout_rate=kwargs.get("dropout_rate", 0)
        )
        self.dense = nn.Linear(att_hidden_units[-1], 1)

    def forward(self, query, keys, keys_lenght, mask=None):
        """
        query: 候选item, [btz, 1, emb_size]
        keys:  历史点击序列, [btz, seq_len, emb_size]
        keys_len: keys的长度, [btz, 1]
        mask: [btz, seq_len]
        """
        btz, seq_len, emb_size = keys.shape

        # 计算注意力分数
        queries = query.expand(-1, seq_len, -1)
        attn_input = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1) # [btz, seq_len, 4*emb_size]
        attn_output = self.dnn(attn_input)  # [btz, seq_len, hidden_units[-1]]
        att_score = self.dense(attn_output) # [btz, seq_len, 1]

        # mask
        if mask is not None:
            keys_mask = mask.unsqueeze(1)
        else:
            keys_mask = torch.arange(seq_len, device=keys.device).repeat(btz, 1)    # [btz, seq_len]
            keys_mask = keys_mask < keys_lenght
            keys_mask = keys_mask.unsqueeze(1)  # [btz, 1, seq_len]

        attn_score = attn_score.transpose(1, 2)  # [btz, 1, seq_len]

        if self.weight_normalization:
            # padding置为-inf，这样softmax后就是0
            attn_score = torch.masked_fill(attn_score, keys_mask.bool(), -1e-7)
            attn_score = F.softmax(attn_score, dim=-1)  # [btz, 1, seq_len]
        else:
            # padding置为0
            attn_score = torch.masked_fill(attn_score, keys_mask.bool(), 0)
        
        if not self.return_score:
            return torch.matmul(attn_score, keys)  # [btz, 1, emb_size]
        return attn_score

class InterestExtractor(nn.Module):
    """ DIEN中的兴趣提取模块
    """
    def __init__(self,
                 input_size,
                 use_neg=False,
                 dnn_hidden_units=[100, 50, 1],
                 init_std=0.001
                 ) -> None:
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=input_size,
            batch_first=True
        )
        if self.use_neg:
            self.auxiliary_net = DNN(
                input_dim=input_size * 2,
                hidden_units=dnn_hidden_units,
                activation="sigmoid",
                init_std=init_std
            )
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, keys, keys_length, neg_keys=None):
        """
        keys:        [btz, seq_len, hdsz]
        keys_length: [btz, 1]
        neg_keys:    [btz, seq_len, hdsz]   
        """
        btz, seq_len, hdsz = keys.shape
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz, 1]

        # keys全部为空
        if keys_length.shape[0] == 0:
            return torch.zeros(btz, hdsz, device=keys.device)
        # 过RNN
        masked_keys = torch.masked_select(keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)    # 去除全为0序列的样本
        packed_keys = pack_padded_sequence(
            masked_keys, 
            lengths=keys_length.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        packed_interests, _ = self.gru(packed_keys)
        interests, _ = pad_packed_sequence(
            packed_interests,
            batch_first=True,
            padding_value=0.0,
            total_length=seq_len
        )
        # 计算auxiliary_loss
        if self.use_neg and neg_keys is not None:
            masked_neg_keys = torch.masked_select(neg_keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)
            aux_loss = self._cal_auxiliary_loss(interests[:, :-1, :], masked_keys[:, 1:, :], 
                                                masked_neg_keys[:, 1:, :], keys_length - 1)
        return interests, aux_loss
    
    def _cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
        """
        states:        [btz, seq_len, hdsz]
        click_seq:     [btz, seq_len, hdsz]   
        noclick_seq:   [btz, seq_len, hdsz]
        keys_length:   [btz, 1]
        """
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz1, 1]

        # keys全部为空
        if keys_length.shape[0] == 0:
            return torch.zeros((1,), device=states.device)
        
        # 去除全为0序列的样本
        btz, seq_len, hdsz = states.shape
        states = torch.masked_select(states, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)
        click_seq = torch.masked_select(click_seq, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)
        noclick_seq = torch.masked_select(noclick_seq, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)

        # 仅对非mask部分计算loss
        mask = torch.arange(seq_len, device=states.device) < keys_length[:, None]
        click_input = torch.cat([states, click_seq], dim=-1)  # [btz, seq_len, hdsz*2]
        noclick_input = torch.cat([states, noclick_seq], dim=-1)  # [btz, seq_len, hdsz*2]
        click_p = self.auxiliary_net(click_input.view(-1, hdsz*2)).view(btz, seq_len)[mask].view(-1, 1)
        noclick_p = self.auxiliary_net(noclick_input.view(-1, hdsz*2)).view(btz, seq_len)[mask].view(-1, 1)
        click_target = torch.ones_like(click_p)
        noclick_target = torch.zeros_like(click_p)

        loss = F.binary_cross_entropy(torch.cat([click_p, noclick_p], dim=0), torch.cat([click_target, noclick_target], dim=0))
        return loss

class AGRUCell(nn.Module):
    """ Attention based GRU (AGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_hh', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, _, i_n = gi.chunk(3, 1)
        h_r, _, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        # update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        hy = (1. - att_score) * hx + att_score * new_state
        return hy

class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        att_score = att_score.view(-1, 1)
        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, gru_type='AGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if gru_type == 'AGRU':
            self.rnn = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.rnn = AUGRUCell(input_size, hidden_size, bias)

    def forward(self, inputs, att_scores=None, hx=None):
        if not isinstance(inputs, PackedSequence) or not isinstance(att_scores, PackedSequence):
            raise NotImplementedError("DynamicGRU only supports packed input and att_scores")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        att_scores, _, _, _ = att_scores

        max_batch_size = int(batch_sizes[0])
        if hx is None:
            hx = torch.zeros(max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)

        outputs = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)

        begin = 0
        for batch in batch_sizes:
            new_hx = self.rnn(
                inputs[begin:begin + batch],
                hx[0:batch],
                att_scores[begin:begin + batch])
            outputs[begin:begin + batch] = new_hx
            hx = new_hx
            begin += batch
        return PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices)
    
class InterestEvolving(nn.Module):
    """DIEN中的兴趣演化模块
    """
    def __init__(self, input_size, gru_type='GRU', use_neg=False, init_std=0.001, 
                 att_hidden_size=(64, 16), att_activation='sigmoid', att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        assert gru_type in {'GRU', 'AIGRU', 'AGRU', 'AUGRU'}, f"gru_type: {gru_type} is not supported"
        self.gru_type = gru_type

        return_score = True
        if gru_type == 'GRU':
            return_score = False
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size, gru_type=gru_type)

        self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size, att_hidden_units=att_hidden_size, att_activation=att_activation,
                                                       weight_normalization=att_weight_normalization, return_score=return_score)

        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    @staticmethod
    def _get_last_state(states, keys_length):
        # states [B, T, H]
        batch_size, max_seq_length, _ = states.size()

        mask = (torch.arange(max_seq_length, device=keys_length.device).repeat(
            batch_size, 1) == (keys_length.view(-1, 1) - 1))

        return states[mask]

    def forward(self, query, keys, keys_length, mask=None):
        """
        query:       [btz, 1, hdsz]
        keys:        [btz, seq_len ,hdsz]
        keys_length: [btz, 1]
        """
        btz, seq_len, hdsz = keys.shape
        smp_mask = keys_length > 0
        keys_length = keys_length[smp_mask]  # [btz1, 1]

        # keys全部为空
        zero_outputs = torch.zeros(btz, hdsz, device=query.device)
        if keys_length.shape[0] == 0:
            return zero_outputs

        query = torch.masked_select(query, smp_mask.view(-1, 1, 1)).view(-1, 1, hdsz)
        keys = torch.masked_select(keys, smp_mask.view(-1, 1, 1)).view(-1, seq_len, hdsz)  # 去除全为0序列的样本

        if self.gru_type == 'GRU':
            packed_keys = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_interests, _ = self.interest_evolution(packed_keys)
            interests, _ = pad_packed_sequence(packed_interests, batch_first=True, padding_value=0.0, total_length=seq_len)
            outputs = self.attention(query, interests, keys_length.unsqueeze(1))  # [btz1, 1, hdsz]
            outputs = outputs.squeeze(1)  # [btz1, hdsz]

        elif self.gru_type == 'AIGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1))  # [btz1, 1, seq_len]
            interests = keys * att_scores.transpose(1,2)  # [btz1, seq_len, hdsz]
            packed_interests = pack_padded_sequence(interests, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            _, outputs = self.interest_evolution(packed_interests)
            outputs = outputs.squeeze(0)  # [btz1, hdsz]

        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            att_scores = self.attention(query, keys, keys_length.unsqueeze(1)).squeeze(1)  # [b, T]
            packed_interests = pack_padded_sequence(keys, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            packed_scores = pack_padded_sequence(att_scores, lengths=keys_length.cpu(), batch_first=True, enforce_sorted=False)
            outputs = self.interest_evolution(packed_interests, packed_scores)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0, total_length=seq_len)
            # pick last state
            outputs = InterestEvolving._get_last_state(outputs, keys_length) # [b, H]
            
        # [b, H] -> [B, H]
        zero_outputs[smp_mask.squeeze(1)] = outputs
        return zero_outputs


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model

    """
    def __init__(self, in_features, layer_num=2, parameterization='vector'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])


    def forward(self, inputs):
        """
        inputs: [btz, in_features]
        """
        x_0 = inputs.unsqueeze(2)  # [btz, in_features, 1]
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                # 这里先计算的后两项 x' * W: [btz, in_features, 1] * [in_features, 1] = [btz, 1, 1]
                # 再计算x0*xl_w [btz, in_features, 1] * [btz, 1, 1] = [btz, in_features, 1]
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)

                # 也可以先计算前两项, 已经验证过两者是相等的
                # xl_w = torch.matmul(x_l, self.kernels[i].T)  # [btz, in_features, in_features]
                # dot_ = torch.matmul(xl_w, x_0)  # [btz, in_features, 1]

                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)  # [btz, in_features]
        return x_l   



