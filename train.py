import logging

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from model import PatterMiningTrainer, MultiValue_PatternClassifier
from dataloader import CustomTabularDataset

parser = ArgumentParser()
parser.add_argument('--train_data', type=str, required=True, help="Training data. This will also be used to define model.")
parser.add_argument('--dim_hidden', type=int, default=32, help='Network hidden dimension')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='gpu', choices=['gpu','cpu'])


def get_model(feat_size_list, args):
    return MultiValue_PatternClassifier(feat_size_list=feat_size_list, args=args)


def train(trainer, dataset, args):
    epoch = args.epoch

    logging.info('Start training')
    for ep in range(epoch):
        print(f'Epoch: {ep}')
        for step, (transaction, label) in enumerate(dataset):
            loss = trainer.update(transaction, label)
            #if step % 20 == 0:
            #    print(f'Step {step}')
            if step % 500 == 0:
                print(f'Epoch: {ep}, step: {step}, loss: {loss}')
    logging.info('Finish training')


def main(args):
    data_path = args.train_data
    device = args.device
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Device set to {device}')

    dataset = CustomTabularDataset(data_path=data_path)
    training_generator = DataLoader(dataset, batch_size=args.batch_size)
    feat_size_list = dataset.get_feature_size_list()
    model = get_model(feat_size_list=feat_size_list, args=args)
    trainer = PatterMiningTrainer(model=model, args=args, device=device)
    train(trainer, training_generator, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)