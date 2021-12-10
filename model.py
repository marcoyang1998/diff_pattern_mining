import torch
import torch.nn as nn
from utils import pattern_init_weight
from abc import ABC, abstractmethod


class BasePatternMiner(nn.Module, ABC):
    def __init__(self):
        super(BasePatternMiner, self).__init__()

    @abstractmethod
    def clamp_weight(self):
        pass

    @abstractmethod
    def get_pattern(self):
        pass


class MultiValuePatternClassifier(nn.Module, BasePatternMiner):
    def __init__(self, feat_size_list, args):
        super(MultiValuePatternClassifier, self).__init__()
        self.feature_value_list = feat_size_list
        self.num_feature = len(feat_size_list)
        self.feature_start_pos_list = []
        pos = 0
        for i in range(self.num_feature):
            self.feature_start_pos_list.append(pos)
            pos += self.feature_value_list[i]

        self.input_linears = nn.ModuleList()
        for i in range(self.num_feature):
            self.input_linears.append(nn.Linear(feat_size_list[i], 1))
        self.input_activation = nn.ReLU()

        self.dim_hidden = args.dim_hidden
        self.embedding = nn.Linear(self.num_feature, self.dim_hidden)
        self.embedding_activation = nn.ReLU()
        self.output_layer = nn.Linear(self.dim_hidden, 1)

        self.init_weight()

    def init_weight(self):
        self.input_linears.apply(pattern_init_weight)
        self.embedding.apply(pattern_init_weight)
        self.output_layer.apply(pattern_init_weight)

    def forward(self, x):
        feat_list = []
        for i in range(self.num_feature):
            feature = x[:, self.feature_start_pos_list[i]:self.feature_start_pos_list[i] + self.feature_value_list[i]]
            feature = self.input_activation(self.input_linears[i](feature))
            feat_list.append(feature)
        x = torch.cat(feat_list, dim=-1)
        x = self.embedding_activation(self.embedding(x))
        x = self.output_layer(x).squeeze(dim=-1)
        return x


class BinaryPatternClassifier(nn.Module, BasePatternMiner):
    def __init__(self, num_feat, args):
        super(BinaryPatternClassifier, self).__init__()
        self.num_feature = num_feat
        self.dim_hidden = args.dim_hidden
        self.linear_encoder = nn.Linear(num_feat, self.dim_hidden)
        self.encoder_activation = nn.ReLU()

        self.classifier = nn.Linear(self.dim_hidden, 1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.encoder_activation(self.linear_encoder(x))
        x = self.classifier(x)

        return x

    def get_pattern(self):
        return self.linear_encoder.weight, self.classifier.weight

    def init_weight(self):
        self.linear_encoder.apply(pattern_init_weight)
        self.classifier.apply(pattern_init_weight)

    def clamp_weight(self):
        with torch.no_grad():
            torch.clamp_(self.linear_encoder.weight, min=0.0, max=1.0)
            torch.clamp_(self.classifier.weight, min=0.0, max=1.0)
            torch.clamp_(self.linear_encoder.bias, max=0.0)


class BinaryPatternEmbedding(nn.Module):
    def __init__(self, num_feat, args):
        super(BinaryPatternEmbedding, self).__init__()
        self.num_feature = num_feat
        self.dim_hidden = args.dim_hidden
        self.linear_encoder = nn.Linear(num_feat, self.dim_hidden)
        self.encoder_activation = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.encoder_activation(self.linear_encoder(x))
        return x

    def init_weight(self):
        self.linear_encoder.apply(pattern_init_weight)

    def get_pattern(self):
        return self.linear_encoder.weight

    def clamp_weight(self):
        with torch.no_grad():
            torch.clamp_(self.linear_encoder.weight, min=0.0, max=1.0)
            torch.clamp_(self.linear_encoder.bias, max=0.0)


def get_model(args, **kwargs):
    if args.dataset_type == 'binary':
        num_feat = kwargs.get('num_feat')
        if args.loss_type == 'classification':
            return BinaryPatternClassifier(args=args, num_feat=num_feat)
        elif args.loss_type == 'contrastive':
            return BinaryPatternEmbedding(args=args, num_feat=num_feat)
    else:
        feat_size_list = kwargs.get('feat_size_list')
        return MultiValuePatternClassifier(args=args, feat_size_list=feat_size_list)