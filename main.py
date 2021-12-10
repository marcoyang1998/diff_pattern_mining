import logging
from argparse import ArgumentParser
from utils import to_device

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trainer import get_trainer
from dataloader import get_dataloader

parser = ArgumentParser()
parser.add_argument('--train_data', type=str, required=True, help="Training data. This will also be used to define model.")
parser.add_argument('--dim_hidden', type=int, default=32, help='Network hidden dimension')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_type', type=str, choices=['binary', 'multivalue'], default='binary')
parser.add_argument('--loss_type', type=str, choices=['classification','contrastive'], default='classification')
parser.add_argument('--device', type=str, default='gpu', choices=['gpu','cpu'])
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--debug_dir', type=str, default='log')
np.set_printoptions(suppress=True)

def train(trainer, dataset, writer, args):
    epoch = args.epoch
    output_json = {}
    logging.info('Start training')
    for ep in range(epoch):
        loss_list = []
        print(f'Epoch: {ep}')
        for step, data in enumerate(dataset):
            data = to_device(data, device=trainer.device)
            loss = trainer.update(*data)
            if step % 500 ==0:
                print(f'Epoch: {ep}, step: {step}, loss: {loss}')

        loss_list.append(loss.cpu().detach().numpy())
        if ep % 10 == 0:
            trainer.print_pattern()
        writer.add_scalar('loss', loss, ep)
    logging.info('Finish training')


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
    model_info = dataset.get_model_info()
    trainer = get_trainer(args=args, device=device, **model_info)
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