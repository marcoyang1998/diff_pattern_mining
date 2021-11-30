import logging
from argparse import ArgumentParser

import torch
import numpy as np
from torch.utils.data import DataLoader


from model import PatterMiningTrainer, get_model
from dataloader import BinaryTabularDataset, MultiValueTabularDataset

parser = ArgumentParser()
parser.add_argument('--train_data', type=str, required=True, help="Training data. This will also be used to define model.")
parser.add_argument('--dim_hidden', type=int, default=32, help='Network hidden dimension')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_type', type=str, choices=['binary','multivalue'], default='binary')
parser.add_argument('--device', type=str, default='gpu', choices=['gpu','cpu'])


def train(trainer, dataset, args):
    epoch = args.epoch
    pattern_list = []
    logging.info('Start training')
    for ep in range(epoch):
        print(f'Epoch: {ep}')
        for step, (transaction, label) in enumerate(dataset):
            loss = trainer.update(transaction, label)
            #if step % 20 == 0:
            #    print(f'Step {step}')
            if step % 500 == 0:
                print(f'Epoch: {ep}, step: {step}, loss: {loss}')
        current_pattern = trainer.get_pattern()
        print(f'Current pattern: {current_pattern}')
        pattern_list.append(current_pattern.cpu().detach().numpy())
    pattern_list = np.array(pattern_list)
    logging.info('Finish training')
    return pattern_list


def main(args):
    data_path = args.train_data
    device = args.device
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Device set to {device}')
    if args.dataset_type == 'binary':
        dataset = BinaryTabularDataset(data_path=data_path)
    else:
        dataset = MultiValueTabularDataset(data_path=data_path)
    with open(data_path.replace('csv','txt'), 'r') as f:
        ground_truth = f.readline()
    ground_truth = list(map(int, ground_truth.split(' ')))
    training_generator = DataLoader(dataset, batch_size=args.batch_size)
    feat_size_list = dataset.get_model_info()
    model = get_model(args=args, **feat_size_list)
    trainer = PatterMiningTrainer(model=model, args=args, device=device)
    pattern_list = train(trainer, training_generator, args)
    print('Ground truth pattern: ', ground_truth)
    print('Learned pattern: ', pattern_list[-1])
    np.save('patterns.npy',pattern_list)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)