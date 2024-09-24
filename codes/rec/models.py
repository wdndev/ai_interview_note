import torch
from torch import nn
from .inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list
from .inputs import combined_dnn_input, create_embedding_matrix, embedding_lookup, maxlen_lookup, split_columns, input_from_feature_columns
from .layers import FM, DNN, PredictionLayer, AttentionSequencePoolingLayer, InterestExtractor, InterestEvolving, CrossNet, ResidualNetwork
from .utils import get_kw

class RecLinear(nn.Module):
    """浅层线性全连接，也就是 Wide&Cross 的Wide部分
    步骤：
    1. Sparse特征分别过embedding, 得到多个[btz, 1, 1]
    2. VarLenSparse过embeddingg+pooling后，得到多个[btz, 1, 1]
    3. Dense特征直接取用, 得到多个[btz, dense_len]
    4. Sparse和VarLenSparse进行cat得到[btz, 1, featnum]，再sum_pooling得到[btz, 1]的输出
    5. Dense特征过[dense_len, 1]的全连接得到[btz, 1]的输出
    6. 两者求和得到最后输出
    
    参数：
        feature_columns: 各个特征的[SparseFeat, VarlenSparseFeat, DenseFeat, ...]的列表
        feature_index: 每个特征在输入tensor X中的列的起止位置索引
        init_std: 初始化标准差
        out_dim: 输出维度，默认为1（回归或二分类）
    """
    def __init__(
            self, 
            feature_columns, 
            feature_index, 
            init_std=1e-4, 
            out_dim=1, 
            **kwargs
        ):
        super(RecLinear, self).__init__()
        self.feature_index = feature_index
        self.out_dim = out_dim
        self.feature_columns = feature_columns
        self.sparse_feature_columns, self.dense_feature_columns, self.varlen_sparse_feature_columns = split_columns(feature_columns)
        
        # 特征embdding字典，{feat_name: nn.Embedding()}
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, out_dim, sparse=False)  # out_dim=1表示线性
        
        if len(self.dense_feature_columns) > 0:
            # 如果存在Dense特征，则初始化一个全连接层的权重
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), out_dim))
            nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        # 获取 Sparse 和 VarLenSparse 特征的嵌入向量，以及 Dense 特征值
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            X, 
            self.feature_columns, 
            self.feature_index, 
            self.embedding_dict
        )

        # 初始化输出张量
        linear_logit = torch.zeros([X.shape[0], self.out_dim], device=X.device)
        
        if len(sparse_embedding_list) > 0:
            # 将所有 Sparse 和 VarLenSparse 特征的嵌入向量拼接，并可能进行加权处理
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)  # [btz, 1, feat_cnt]
            if sparse_feat_refine_weight is not None:  # 加权
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            # 对拼接后的特征进行求和，得到每个样本的线性组合
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            # 对Dense特征通过全连接层得到其线性组合
            dense_value_logit = torch.cat(dense_value_list, dim=-1).float().matmul(self.weight)
            linear_logit += dense_value_logit
        
        # 返回线性部分的输出
        return linear_logit


class RecBase(nn.Module):
    def __init__(
            self,
            linear_feature_columns, 
            dnn_feature_columns, 
            l2_reg_linear=1e-5, 
            l2_reg_embedding=1e-5,
            init_std=1e-4, 
            out_dim=1, 
            **kwargs
        ):
        super(RecBase, self).__init__()
        self.dnn_feature_columns = dnn_feature_columns
        self.aux_loss = 0  # 目前只看到dien里面使用

        # feat_name到col_idx的映射, eg: {'age':(0,1),...}
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # 为 SparseFeat 和 VarLenSparseFeat 特征创建 embedding
        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False)
        self.linear_model = RecLinear(
            linear_feature_columns, 
            self.feature_index, 
            out_dim=out_dim, 
            **kwargs
        )

        # l1和l2正则
        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        # 输出层
        self.out = PredictionLayer(out_dim, **kwargs)

    def compute_input_dim(self, feature_columns, feature_names=[('sparse', 'var_sparse', 'dense')], feature_group=False):
        '''计算输入维度和，Sparse/VarlenSparse的embedding_dim + Dense的dimesion
        '''
        def get_dim(feat):
            if isinstance(feat, DenseFeat):
                return feat.dimension
            elif feature_group:
                return 1
            else:
                return feat.embedding_dim

        feature_col_groups = split_columns(feature_columns, feature_names)
        input_dim = 0
        for feature_col in feature_col_groups:
            if isinstance(feature_col, list):
                for feat in feature_col:
                    input_dim += get_dim(feat)
            else:
                input_dim += get_dim(feature_col)
                    
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        """记录需要正则的参数项
        """
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        # 对于一个Parameter对象，将其放入一个列表中以保持与get_regularization_loss()方法的兼容性
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        # 对于生成器、过滤器和ParameterList，转换为张量列表以避免错误
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        """计算正则损失
        """
        total_reg_loss = 0
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = split_columns(feature_columns, ['sparse', 'var_sparse'])
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]

class DeepCrossing(RecBase):
    """DeepCrossing的实现
    和Wide&Deep相比，去掉Wide部分，DNN部分换成残差网络，模型结构简单
    [1] [ACM 2016] Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features (https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)
    """
    def __init__(
            self, 
            linear_feature_columns, 
            dnn_feature_columns, 
            dnn_hidden_units=(256, 128),
            l2_reg_linear=1e-5, 
            l2_reg_embedding=1e-5, 
            l2_reg_dnn=0, 
            init_std=1e-4,
            dnn_dropout=0, 
            dnn_activation='relu', 
            dnn_use_bn=False, 
            out_dim=1, 
            **kwargs
        ):
        super(DeepCrossing, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                           l2_reg_embedding=l2_reg_embedding, init_std=init_std, out_dim=out_dim, **kwargs)
        del self.linear_model
        assert len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.dnn = ResidualNetwork(
            input_dim, 
            dnn_hidden_units, 
            activation=dnn_activation, 
            dropout_rate=dnn_dropout, 
            use_bn=dnn_use_bn, 
            init_std=init_std
        )
        self.dnn_linear = nn.Linear(input_dim, 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        # 离散变量过embedding，连续变量保留原值
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.dnn_feature_columns, self.feature_index, self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit

        y_pred = self.out(logit)

        return y_pred

class NeuralCF(RecBase):
    """NeuralCF的实现，用于召回
    输入是(user, item)的数据对， 两者的的embedding_dim需要一致
    [1] [WWW 2017] Neural Collaborative Filtering (https://arxiv.org/pdf/1708.05031.pdf)
    """
    def __init__(
            self, 
            dnn_feature_columns, 
            dnn_hidden_units=(256, 128), 
            dnn_emd_dim=4,
            l2_reg_linear=1e-5, 
            l2_reg_embedding=1e-5, 
            l2_reg_dnn=0, 
            init_std=1e-4,
            dnn_dropout=0, 
            dnn_activation='relu', 
            dnn_use_bn=False, 
            out_dim=1
        ):
        super(NeuralCF, self).__init__([], dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                       l2_reg_embedding=l2_reg_embedding, init_std=init_std, out_dim=out_dim)
        assert len(dnn_feature_columns) == 2
        assert dnn_feature_columns[0].embedding_dim == dnn_feature_columns[1].embedding_dim

        # DNN部分
        # 从feature_columns解析处的self.embedding_dict作为mf_embedding_dict，这里是生成mlp的嵌入
        self.dnn_embedding_dict = create_embedding_matrix(
            dnn_feature_columns, 
            init_std, 
            out_dim=dnn_emd_dim, 
            sparse=False
        )  # out_dim=1表示线性
        self.dnn = DNN(
            self.compute_input_dim(dnn_feature_columns), 
            dnn_hidden_units, 
            activation=dnn_activation, 
            dropout_rate=dnn_dropout, 
            use_bn=dnn_use_bn, 
            init_std=init_std
        )
        self.dnn_linear = nn.Linear(
            dnn_feature_columns[0].embedding_dim + dnn_hidden_units[-1], 
            1, 
            bias=False
        )
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        ''' X: [btz, 2]
        '''
        assert X.shape[1] == 2, 'NeuralCF accept (user, item) pair inputs'

        # MF(矩阵因子分解)部分
        sparse_embedding_list, _ = input_from_feature_columns(
            X, 
            self.dnn_feature_columns, 
            self.feature_index, 
            self.embedding_dict
        )
        mf_vec = torch.mul(
            sparse_embedding_list[0], 
            sparse_embedding_list[1]
        ).squeeze(1)

        # DNN部分
        sparse_embedding_list, _ = input_from_feature_columns(
            X, 
            self.dnn_feature_columns, 
            self.feature_index, 
            self.dnn_embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, [])  # [btz, sparse_feat_cnt*emb_size]
        dnn_vec = self.dnn(dnn_input)
        
        # 合并两个
        vector = torch.cat([mf_vec, dnn_vec], dim=-1)
        
        # Linear部分
        logit = self.dnn_linear(vector)
        return self.out(logit)


class DeepFM(RecBase):
    """DeepFM的实现
        Reference: [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
    """
    def __init__(
            self, 
            linear_feature_columns, 
            dnn_feature_columns, 
            use_fm=True,        # 是否启用Factorization Machine (FM) 部分
            dnn_hidden_units=(256, 128),
            l2_reg_linear=1e-5, 
            l2_reg_embedding=1e-5, 
            l2_reg_dnn=0, 
            init_std=1e-4,
            dnn_dropout=0, 
            dnn_activation='relu', 
            dnn_use_bn=False, 
            out_dim=1, 
            **kwargs
        ):
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, out_dim=out_dim, **kwargs)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(
                self.compute_input_dim(dnn_feature_columns), 
                dnn_hidden_units, 
                activation=dnn_activation, 
                dropout_rate=dnn_dropout, 
                use_bn=dnn_use_bn, 
                init_std=init_std
            )
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 
                out_dim, 
                bias=False
            )
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        # 离散变量过embedding，连续变量保留原值
        sparse_embedding_list, dense_value_list = input_from_feature_columns(
            X, 
            self.dnn_feature_columns, 
            self.feature_index, 
            self.embedding_dict
        )
        logit = self.linear_model(X)  # [btz, out_dim]

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)  # [btz, feat_cnt, emb_size]
            # FM仅对离散特征进行交叉
            logit += self.fm(fm_input)  # [btz, out_dim]

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred

class WideDeep(RecBase):
    """WideDeep的实现
    Wide部分是SparseFeat过embedding, VarlenSparseFeat过embedding+pooling, Dense特征直接取用
    Deep部分所有特征打平[btz, sparse_feat_cnt*emb_size+dense_feat_cnt]过DNN
    Reference: [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
    """
    def __init__(
            self, 
            linear_feature_columns, 
            dnn_feature_columns, 
            dnn_hidden_units=(256, 128),
            l2_reg_linear=1e-5, 
            l2_reg_embedding=1e-5, 
            l2_reg_dnn=0, 
            init_std=1e-4, 
            dnn_dropout=0, 
            dnn_activation='relu', 
            dnn_use_bn=False, 
            out_dim=1, 
            **kwargs
        ):
        super(WideDeep, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, out_dim=out_dim, **kwargs)

        if len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0:
            self.dnn = DNN(
                self.compute_input_dim(dnn_feature_columns), 
                dnn_hidden_units, 
                activation=dnn_activation, 
                dropout_rate=dnn_dropout, 
                use_bn=dnn_use_bn, 
                init_std=init_std
            )
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        # SparseFeat和VarLenSparseFeat生成Embedding，VarLenSparseFeat要过Pooling, DenseFeat直接从X中取用
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.dnn_feature_columns, self.feature_index, self.embedding_dict)
        logit = self.linear_model(X)  # [btz, 1]

        if hasattr(self, 'dnn') and hasattr(self, 'dnn_linear'):
            # 所有特征打平并concat在一起，[btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


class DeepCross(WideDeep):
    """Deep&Cross
    和Wide&Deep相比，是用CrossNet替换了linear_model
    [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12. (https://arxiv.org/abs/1708.05123)
    [2] Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020. (https://arxiv.org/abs/2008.13535)
    """
    def __init__(
            self, 
            linear_feature_columns, 
            dnn_feature_columns, 
            cross_num=2, 
            cross_parameterization='vector',
            dnn_hidden_units=(256, 128), 
            l2_reg_linear=1e-5, 
            l2_reg_embedding=1e-5, 
            l2_reg_cross=1e-5,
            l2_reg_dnn=0, 
            init_std=0.0001, 
            dnn_dropout=0, 
            dnn_activation='relu', 
            dnn_use_bn=False, 
            out_dim=1, 
            use_linear=False, 
            **kwargs
        ):
        super(DeepCross, self).__init__(linear_feature_columns, dnn_feature_columns, **get_kw(DeepCross, locals()))

        # 默认应该不使用linear_model
        if not use_linear:
            del self.linear_model

        dnn_linear_in_feature = 0
        if len(dnn_hidden_units) > 0:
            dnn_linear_in_feature += dnn_hidden_units[-1]
        if cross_num > 0:
            dnn_linear_in_feature += self.compute_input_dim(dnn_feature_columns)
        
        if dnn_linear_in_feature > 0:
            self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)
        self.crossnet = CrossNet(in_features=self.compute_input_dim(dnn_feature_columns),
                                 layer_num=cross_num, parameterization=cross_parameterization)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.dnn_feature_columns, self.feature_index, self.embedding_dict)

        # 线性部分，默认不使用
        logit = self.linear_model(X) if hasattr(self, 'linear_model') else 0 # [btz, 1]

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]

        # CrossNetwork
        stack_out = [self.crossnet(dnn_input)]

        # Deep Network
        if hasattr(self, 'dnn'):
            stack_out.append(self.dnn(dnn_input))
        stack_out = torch.cat(stack_out, dim=-1)

        if hasattr(self, 'dnn_linear'):
            logit += self.dnn_linear(stack_out)

        # Out
        y_pred = self.out(logit)
        return y_pred

class DIN(RecBase):
    """Deep Interest Network实现
    """
    def __init__(
            self, 
            dnn_feature_columns, 
            item_history_list, 
            dnn_hidden_units=(256, 128),
            att_hidden_units=(64, 16), 
            att_activation='Dice', 
            att_weight_normalization=False,
            l2_reg_embedding=1e-5, 
            l2_reg_dnn=0, 
            init_std=1e-4,
            dnn_dropout=0, 
            dnn_activation='relu', 
            dnn_use_bn=False, 
            out_dim=1, 
            **kwargs
        ):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_embedding=l2_reg_embedding, init_std=init_std, out_dim=out_dim, **kwargs)
        del self.linear_model  # 删除不必要的网络结构
        
        self.sparse_feature_columns, self.dense_feature_columns, self.varlen_sparse_feature_columns = split_columns(dnn_feature_columns)
        self.item_history_list = item_history_list

        # 把varlen_sparse_feature_columns分解成hist、neg_hist和varlen特征
        # 其实是DIEN的逻辑（为了避免多次执行），DIN中少了neg模块，DIEN是在deepctr是在forward中会重复执行多次
        self.history_feature_names = list(map(lambda x: "hist_"+x, item_history_list))
        self.neg_history_feature_names = list(map(lambda x: "neg_" + x, self.history_feature_names))
        self.history_feature_columns = []
        self.neg_history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_feature_names:
                self.history_feature_columns.append(fc)
            elif feature_name in self.neg_history_feature_names:
                self.neg_history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        # Attn模块
        att_emb_dim = self._compute_interest_dim()
        self.attention = AttentionSequencePoolingLayer(
            att_hidden_units=att_hidden_units, 
            embedding_dim=att_emb_dim, 
            att_activation=att_activation,
            return_score=False, 
            supports_masking=False, 
            weight_normalization=att_weight_normalization
        )

        # DNN模块
        self.dnn = DNN(
            self.compute_input_dim(dnn_feature_columns), 
            dnn_hidden_units, 
            activation=dnn_activation, 
            dropout_rate=dnn_dropout, 
            use_bn=dnn_use_bn, 
            init_std=init_std
        )
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)


    def forward(self, X):
        # 过embedding
        emb_lists, query_emb, keys_emb, keys_length, deep_input_emb = self._get_emb(X)

        # 获取变长稀疏特征pooling的结果， [[btz, 1, emb_size]
        sequence_embed_dict = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_varlen_feature_columns)
        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index, self.sparse_varlen_feature_columns)
        
        # Attn部分
        hist = self.attention(query_emb, keys_emb, keys_length)  # [btz, 1, hdsz]

        # dnn部分
        dnn_input_emb_list = emb_lists[2]
        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat([deep_input_emb, hist], dim=-1)  # [btz, 1, hdsz]
        dnn_input = combined_dnn_input([deep_input_emb], emb_lists[-1])  # [btz, hdsz]
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        # 输出
        y_pred = self.out(dnn_logit)

        return y_pred
        
    def _get_emb(self, X):
        # 过embedding，这里改造embedding_lookup使得只经过一次embedding, 加快训练速度
        # query_emb_list     [[btz, 1, emb_size], ...]
        # keys_emb_list      [[btz, seq_len, emb_size], ...]
        # dnn_input_emb_list [[btz, 1, emb_size], ...]
        return_feat_list = [self.item_history_list, self.history_feature_names, [fc.name for fc in self.sparse_feature_columns]]
        emb_lists = embedding_lookup(X, self.embedding_dict, self.feature_index, self.dnn_feature_columns, return_feat_list=return_feat_list)
        query_emb_list, keys_emb_list, dnn_input_emb_list = emb_lists
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in self.dense_feature_columns]
        emb_lists.append(dense_value_list)

        query_emb = torch.cat(query_emb_list, dim=-1)  # [btz, 1, hdsz]
        keys_emb = torch.cat(keys_emb_list, dim=-1)  # [btz, 1, hdsz]
        keys_length = maxlen_lookup(X, self.feature_index, self.history_feature_names)  # [btz, 1]
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)  # [btz, 1, hdsz]
        return emb_lists, query_emb, keys_emb, keys_length, deep_input_emb

    def _compute_interest_dim(self):
        """计算兴趣网络特征维度和
        """
        dim_list = [feat.embedding_dim for feat in self.sparse_feature_columns if feat.name in self.item_history_list]
        return sum(dim_list)


class DIEN(DIN):
    """ Deep Interest Evolution Network
    """
    def __init__(
            self, 
            dnn_feature_columns, 
            item_history_list, 
            gru_type="GRU", 
            use_negsampling=False, 
            alpha=1.0, 
            dnn_use_bn=False, 
            dnn_hidden_units=(256, 128), 
            dnn_activation='relu', 
            att_hidden_units=(64, 16), 
            att_activation="relu", 
            att_weight_normalization=True, 
            l2_reg_embedding=1e-6, 
            l2_reg_dnn=0, 
            dnn_dropout=0, 
            init_std=0.0001, 
            out_dim=1, 
            **kwargs
        ):
        super(DIEN, self).__init__(dnn_feature_columns, item_history_list, dnn_hidden_units, att_hidden_units, att_activation, att_weight_normalization, 
                                   l2_reg_embedding, l2_reg_dnn, init_std, dnn_dropout, dnn_activation, dnn_use_bn, out_dim, **kwargs)
        del self.attention
        self.alpha = alpha

        # 兴趣提取层
        input_size = self._compute_interest_dim()
        self.interest_extractor = InterestExtractor(
            input_size=input_size, 
            use_neg=use_negsampling, 
            init_std=init_std
        )

        # 兴趣演变层
        self.interest_evolution = InterestEvolving(
            input_size=input_size, 
            gru_type=gru_type, 
            use_neg=use_negsampling, 
            init_std=init_std,
            att_hidden_size=att_hidden_units, 
            att_activation=att_activation, 
            att_weight_normalization=att_weight_normalization
        )
        
        # DNN
        dnn_input_size = self.compute_input_dim(
            dnn_feature_columns, 
            [('sparse', 'dense')]
        ) + input_size
        self.dnn = DNN(
            dnn_input_size, 
            dnn_hidden_units, 
            activation=dnn_activation, 
            dropout_rate=dnn_dropout, 
            use_bn=dnn_use_bn, 
            init_std=init_std
        )

    def forward(self, X):
        # 过embedding
        emb_lists, query_emb, keys_emb, keys_length, deep_input_emb = self._get_emb(X)
        neg_keys_emb_list = embedding_lookup(
            X, 
            self.embedding_dict, 
            self.feature_index, 
            self.dnn_feature_columns, 
            return_feat_list=self.neg_history_feature_names
        )
        neg_keys_emb = torch.cat(neg_keys_emb_list, dim=-1)  # [btz, 1, hdsz]

        # 过兴趣提取层
        # input shape: [btz, seq_len, hdsz],  [btz, 1], [btz, seq_len, hdsz]
        # masked_interest shape: [btz, seq_len, hdsz]
        masked_interest, aux_loss = self.interest_extractor(
            keys_emb, 
            keys_length, 
            neg_keys_emb
        )
        self.add_auxiliary_loss(aux_loss, self.alpha)

        # 过兴趣演变层
        hist = self.interest_evolution(query_emb, masked_interest, keys_length)  # [btz, hdsz]

        # dnn部分
        deep_input_emb = torch.cat([deep_input_emb.squeeze(1), hist], dim=-1)  # [btz, hdsz]
        dnn_input = combined_dnn_input([deep_input_emb], emb_lists[-1])  # [btz, hdsz]
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        # 输出
        y_pred = self.out(dnn_logit)

        return y_pred





