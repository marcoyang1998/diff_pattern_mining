import torch
import torch.nn as nn
from torch import optim


class MultiValuePatternClassifier(nn.Module):
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
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.input_linears.apply(_init_weight)
        self.embedding.apply(_init_weight)
        self.output_layer.apply(_init_weight)

    def forward(self, x):
        feat_list = []
        for i in range(self.num_feature):
            feature = x[:, self.feature_start_pos_list[i]:self.feature_start_pos_list[i] + self.feature_value_list[i]]
            feature = self.input_activation(self.input_linears[i](feature))
            feat_list.append(feature)
        x = torch.cat(feat_list, dim=-1)
        x = self.embedding_activation(self.embedding(x))
        #x = torch.log_softmax(self.output_layer(x), dim=1)
        x = self.output_layer(x).squeeze(dim=-1)
        return x


class BinaryPatternClassifier(nn.Module):
    def __init__(self, num_feat, args):
        super(BinaryPatternClassifier, self).__init__()
        self.num_feature = num_feat
        self.dim_hidden = args.dim_hidden
        self.linear_encoder = nn.Linear(num_feat, self.dim_hidden)
        self.encoder_activation = nn.ReLU()

        self.classifier = nn.Linear(self.dim_hidden, 1)

    def forward(self, x):
        x = self.encoder_activation(self.linear_encoder(x))
        x = self.classifier(x)

        return x

    def get_pattern(self):
        return self.linear_encoder.weight

    def init_weight(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.linear_encoder.apply(_init_weight)
        self.classifier.apply(_init_weight)


class PatterMiningTrainer(nn.Module):
    def __init__(self, model: nn.Module, args, device='cpu'):
        super(PatterMiningTrainer, self).__init__()
        self.model = model
        self.epoch = 0
        self.optim = optim.SGD(model.parameters(), lr=args.lr)
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

        self.device = device
        self.model.to(device)

    def forward(self, x):
        x.to(self.device)
        out = self.model.forward(x).squeeze(dim=-1)
        return out

    def update(self, x, label):
        label.to(self.device)
        out = self.forward(x)
        loss = self.loss(out, label.float())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

    def get_pattern(self):
        return self.model.get_pattern()


def get_model(args, **kwargs):
    if args.dataset_type == 'binary':
        num_feat = kwargs.get('num_feat')
        return BinaryPatternClassifier(args=args, num_feat=num_feat)
    else:
        feat_size_list = kwargs.get('feat_size_list')
        return MultiValuePatternClassifier(args=args, feat_size_list=feat_size_list)