#!/usr/bin/env python
import argparse
from cycle_gan import CycleGan


parser = argparse.ArgumentParser()

parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset', nargs='?', default='wat_mai_amataros', type=str)
parser.add_argument('--train_folder', nargs='?', default='train', type=str)
parser.add_argument('--A', nargs='?', default='morning', type=str)
parser.add_argument('--B', nargs='?', default='cloudy', type=str)
parser.add_argument('--test_folder', nargs='?', default='test', type=str)
parser.add_argument('--in_channels', nargs='?', default=3, type=int)
parser.add_argument('--batch_size', nargs='?', default=7, type=int)
parser.add_argument('--gen_filters', nargs='?', default=256, type=int)
parser.add_argument('--disc_filters', nargs='?', default=512, type=int)
parser.add_argument('--res_blocks', nargs='?', default=9, type=int)
parser.add_argument('--img_input_size', nargs='?', default=270, type=int)
parser.add_argument('--img_output_size', nargs='?', default=128, type=int)
parser.add_argument('--lr_disc', nargs='?', default=1e-4, type=float)
parser.add_argument('--lr_gen', nargs='?', default=1e-4, type=float)
parser.add_argument('--train_epoch', nargs='?', default=5, type=int)
parser.add_argument('--lr_cycle_mult', nargs='?', default=2.0, type=float)
parser.add_argument('--cycle_loss_A', nargs='?', default=10.0, type=float)
parser.add_argument('--cycle_loss_B', nargs='?', default=10.0, type=float)
parser.add_argument('--similar_distance', nargs='?', default=10, type=int)
parser.add_argument('--beta1', nargs='?', default=.5, type=float)
parser.add_argument('--beta2', nargs='?', default=.999, type=float)
parser.add_argument('--gen_layers', nargs='?', default=2, type=int)
parser.add_argument('--disc_layers', nargs='?', default=3, type=int)
parser.add_argument('--ids_a', type=int, nargs='+', default=[10, 20])
parser.add_argument('--ids_b', type=int, nargs='+', default=[10, 20])
parser.add_argument('--save_root', nargs='?', default='wat_mai_amataros_train', type=str)
params = vars(parser.parse_args())

if __name__ == '__main__':
    lgtSwap = CycleGan(params)
    lgtSwap.train()
