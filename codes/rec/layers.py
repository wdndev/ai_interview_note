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
            
        )



