from collections import namedtuple, OrderedDict, defaultdict
import torch
from torch import nn
from .layers import SequencePoolingLayer
import numpy as np
import pandas as pd

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name', 'group_name'])):
    """离散特征
    """
    def __new__(
            cls,
            name, 
            vocabulary_size, 
            embedding_dim=4, 
            use_hash=False, 
            dtype="int32", 
            embedding_name=None,
            group_name=DEFAULT_GROUP_NAME
        ):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print("[WARNING] Feature Hashing on the fly currently is not supported in torch version")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype, embedding_name, group_name)
    
    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['sparsefeat', 'maxlen', 'pooling', 'length_name'])):
    """变长离散特征
    """
    def __new__(
            cls, 
            sparsefeat, 
            maxlen, 
            pooling='mean', 
            length_name=None
        ):
            return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, pooling, length_name)
    
    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    """连续特征
    """
    def __new__(
            cls, 
            name, 
            dimension=1, 
            dtype="float32"
        ):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


def get_feature_names(feature_columns):
    """获取特征名称
    """
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns):
    """ feat_name到col_range之间的映射
    """
    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def build_input_array(inputs, feature_columns, target=None):
    """根据特征的顺序组装成tensor
    """
    train_y = None
    if isinstance(inputs, pd.DataFrame):
        if target:
            train_y = inputs[target].values
        inputs = {col: np.array(values) for col, values in inputs.to_dict(orient='list').items()}
        
    feature_index = build_input_features(feature_columns)
    train_X = [inputs[feature] for feature in feature_index]
    for i in range(len(train_X)):
        if len(train_X[i].shape) == 1:
            train_X[i] = np.expand_dims(train_X[i], axis=1)
    train_X = np.concatenate(train_X, axis=-1)
    
    if train_y is not None:
        return train_X, train_y
    elif target:
        train_y = inputs[target]
        return train_X, train_y
    else:
        return train_X
    

def combined_dnn_input(sparse_embedding_list, dense_value_list):
    """合并sparse和dense
    """
    res = []
    if len(sparse_embedding_list) > 0:
        # 将离散特征拼接并展平
        res.append(torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1))
    if len(dense_value_list) > 0:
        # 将连续特征拼接并展平
        res.append(torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1))

    if res:
        return torch.cat(res, dim=-1)
    else:
        raise NotImplementedError


def create_embedding_matrix(
        feature_columns, 
        init_std=1e-4, 
        out_dim=0, 
        sparse=False
    ):
    """ 为Sparse, VarLenSparse进行embedding
    参数:
        feature_columns: 所有特征列
        init_std: 初始化标准差
        out_dim: Embedding的输出维度是否自定义，默认为0即使用用户自己的
        sparse: 是否稀疏

    返回:
        {embedding_name: nn.EmbeddingBag}
    """
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
    var_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []
    # embedding字典
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if out_dim == 0 else out_dim, sparse=sparse) 
        for feat in sparse_feature_columns+var_sparse_feature_columns}
    )
    # 初始化权重
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict


def embedding_lookup(
        X, 
        embedding_dict, 
        feature_index, 
        sparse_feature_columns, 
        return_feat_list=[], 
        return_dict=False
    ):
    """查找离散特征的嵌入向量
    
    参数:
        X: 输入数据
        embedding_dict: 特征对应的嵌入字典
        feature_index: 特征对应的col区间
        sparse_feature_columns: 离散特征列
        return_feat_list: 需要返回的特征列表
        return_dict: 是否返回字典形式
    """

    multi = False
    if len(return_feat_list) == 0:
        return_feat_list = [fc.name for fc in sparse_feature_columns]
        return_feat_list_flat = [fc.name for fc in sparse_feature_columns]
    elif isinstance(return_feat_list[0], (list, tuple)):
        # 嵌套一层
        return_feat_list_flat = [j for i in return_feat_list for j in i]
        multi = True
    else:
        return_feat_list_flat = return_feat_list

    # 对满足筛选条件的过embedding
    embedding_vec_dict = {}
    for fc in sparse_feature_columns:
        feature_name = fc.name
        if feature_name in return_feat_list_flat:
            lookup_idx = np.array(feature_index[feature_name])
            emb = embedding_dict[fc.embedding_name](X[:, lookup_idx[0]:lookup_idx[1]].long())
            embedding_vec_dict[feature_name] = emb
    
    if return_dict:
        return embedding_vec_dict
    elif multi:
        return [[embedding_vec_dict[j] for j in i] for i in return_feat_list]
    else:
        return [embedding_vec_dict[i] for i in return_feat_list]


def get_varlen_pooling_list(
        embedding_dict, 
        features, 
        feature_index, 
        varlen_sparse_feature_columns
    ):
    """获取变长稀疏特征pooling的结果
    参数:
        embedding_dict: {feat_name: input_embedding, ...}  [btz, seq_len, emb_size]
        features: [btz, seq_len]
    """
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.name]
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0
            emb = SequencePoolingLayer(mode=feat.pooling, support_masking=True)([seq_emb, seq_mask])
        else:
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False)([seq_emb, seq_length])
            
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def maxlen_lookup(X, feature_index, col_name, padding=0):
    """计算序列长度
    参数:
        X: 输入数据
        feature_index: 特征索引
        col_name: 列名
        padding: 填充值，默认为0
    """
    lookup_idx = np.array(feature_index[col_name[0]])
    max_len = X[:, lookup_idx[0]:lookup_idx[1]].ne(padding)
    return torch.sum(max_len.long(), dim=-1, keepdim=True)  # [btz, 1]


def split_columns(feature_columns, select_columns=('sparse', 'dense', 'var_sparse')):
    """区分各类特征，因为使用比较多，所以提取出来

    参数:
        feature_columns: 所有特征列
        select_columns: 需要选择的特征类型
    """
    select_columns = [select_columns] if isinstance(select_columns, str) else select_columns
    columns_map = {'sparse': SparseFeat, 'var_sparse': VarLenSparseFeat, 'dense': DenseFeat}

    res = []
    for col in select_columns:
        if isinstance(col, str):
            assert col in columns_map, 'select_columns args illegal'
            col_type = columns_map[col]
        elif isinstance(col, (tuple, list)):
             col_type = tuple([columns_map[item] for item in col])
        else:
            raise ValueError('select_columns args illegal')

        res.append(list(filter(lambda x: isinstance(x, col_type), feature_columns)) if len(feature_columns) else [])
    
    return res[0] if len(res) == 1 else res


def input_from_feature_columns(
        X, 
        feature_columns, 
        feature_index, 
        embedding_dict, 
        support_dense=True
    ):
    """ 根据特征列生成输入特征
        SparseFeat和VarLenSparseFeat生成Embedding，VarLenSparseFeat要过Pooling, DenseFeat直接从X中取用
    
    参数:
        X: 输入数据
        feature_columns: 特征列
        feature_index: 特征索引
        embedding_dict: 嵌入字典
        support_dense: 是否支持连续特征
    """
    # 区分各类特征
    sparse_feature_columns, dense_feature_columns, varlen_sparse_feature_columns = split_columns(feature_columns)

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    # 离散特征过embedding, [(btz,1,out_dim), (btz,1,out_dim), ...]
    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for feat in sparse_feature_columns]

    # 序列离散特征过embedding, {feat_name: (btz, seq_len, out_dim)}
    # 过pooling  [(btz,1,out_dim), (btz,1,out_dim), ...]
    sequence_embed_dict = embedding_lookup(X, embedding_dict, feature_index, varlen_sparse_feature_columns, return_dict=True)
    varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index, varlen_sparse_feature_columns)
    
    # 连续特征直接保留 [(btz, dense_len), (btz, dense_len)]
    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list


class TensorDataset(torch.utils.data.TensorDataset):
    '''继承官方的TensorDataset, 添加指定tensor类型和device功能，防止全部数据放到gpu显存过大
    '''
    def __init__(self, *tensors: torch.Tensor, device=None) -> None:
        super().__init__(*tensors)
        self.device = device
    
    def __getitem__(self, index):
        batch = tuple(tensor[index] for tensor in self.tensors)
        if self.device:
            batch = tuple(tensor.to(self.device) for tensor in batch)
        return batch


def collate_fn_device(device):
    '''如果使用自定义TensorDataset，to(device)太耗时，因此可使用collate_fn_device来把tensor转移到device上
    '''
    def collate_fn(batch):
        X, y = map(list, zip(*batch))
        return torch.stack(X, dim=0).to(device), torch.stack(y, dim=0).to(device)
    return collate_fn