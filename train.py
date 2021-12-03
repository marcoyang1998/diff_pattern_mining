import logging
import os
from argparse import ArgumentParser

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import PatterMiningClassifier, PatternMininingContrastiveTrainer, get_model
from dataloader import get_dataloader

parser = ArgumentParser()
parser.add_argument('--train_data', type=str, required=True, help="Training data. This will also be used to define model.")
parser.add_argument('--dim_hidden', type=int, default=32, help='Network hidden dimension')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_type', type=str, choices=['binary','multivalue'], default='binary')
parser.add_argument('--loss_type', type=str, choices=['classification','contrastive'], default='classification')
parser.add_argument('--device', type=str, default='gpu', choices=['gpu','cpu'])
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--debug_dir', type=str, default='log')


def train(trainer, dataset, writer, args):
    epoch = args.epoch
    pattern_list = []
    logging.info('Start training')
    for ep in range(epoch):
        loss_list = []
        print(f'Epoch: {ep}')
        for step, data in enumerate(dataset):
            #transaction = transaction.to(trainer.device)
            #label = label.to(trainer.device)
            loss = trainer.update(*data)
            #if step % 20 == 0:
            #    print(f'Step {step}')
            if step % 500 == 0:
                print(f'Epoch: {ep}, step: {step}, loss: {loss}')
        loss_list.append(loss.cpu().detach().numpy())
        print(f'Epoch {ep}: loss = {np.array(loss_list).mean()}')
        writer.add_scalar('loss', loss, ep)
        current_pattern = trainer.get_pattern()
        print(f'Current pattern: {current_pattern}')
        #pattern_list.append(current_pattern.cpu().detach().numpy())
    #pattern_list = np.array(pattern_list)
    logging.info('Finish training')
    #return pattern_list


def main(args):
    data_path = args.train_data
    device = args.device
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Device set to {device}')

    dataset = get_dataloader(args)
    with open(data_path.replace('csv','txt'), 'r') as f:
        ground_truth = f.readline()
    ground_truth = list(map(int, ground_truth.split(' ')))

    training_generator = DataLoader(dataset, batch_size=args.batch_size)
    feat_size_list = dataset.get_model_info()
    model = get_model(args=args, **feat_size_list)
    trainer = PatternMininingContrastiveTrainer(model=model, args=args, device=device)
    writer = SummaryWriter(args.debug_dir)
    train(trainer, training_generator, writer, args)
    #print('Ground truth pattern: ', ground_truth)
    #print('Learned pattern: ', pattern_list[-1])
    #print('Pattern weight: ', pattern_weight)
    #np.save(os.path.join(args.output_dir, 'patterns.npy'), pattern_list)
    #np.save(os.path.join(args.output_dir, 'patterns_weight.npy'), pattern_weight)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)