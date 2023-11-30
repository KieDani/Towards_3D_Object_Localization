import os
import argparse
parser = argparse.ArgumentParser(description='Start a training.')
parser.add_argument('--gpu', type=int, default=1, help='GPU number')
parser.add_argument('--env', type=str, default=None, help='Training dataset. Allowed values are SD-S, SD-M, SD-L, RD')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--lossmode', type=str, default='2Dpred_nograd', help='Loss mode. Allowed values are 2Dgt, 2Dpred_nograd, 3Dgt')
parser.add_argument('--sin', type=str, default='resnet34', help='Name of backbone network. Allowed values are resnet34, convnext, convnextv2')
parser.add_argument('--forec', type=str, default='exactNDE', help='Forecaster title. Use default value!')
parser.add_argument('--seed', type=int, default=42, help='Seed')
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from general.train import train
from general.config import MyConfig


def paper_general(sin_title, forec_title, lossmode, environment_name, lr, seed):
    timesteps = [2, 4, 6] if 'realball' in environment_name else [4, 8, 15]
    config = MyConfig(lr_coord=lr, timesteps=timesteps, loss_coord_title='L1', forec_title=forec_title,
                      sin_title=sin_title, environment_name=environment_name,
                      folder=f'review', lossmode=lossmode)
    config.seed = seed
    if environment_name == 'parcour_singleenv_multicam':
        config.NUM_EPOCHS = 200
    elif environment_name == 'parcour_multienv_multicam':
        config.NUM_EPOCHS = 100
    else:
        config.NUM_EPOCHS = 400
    train(config)


if __name__ == '__main__':
    assert args.sin in ['resnet34', 'convnext', 'convnextv2']
    assert args.lossmode in ['2Dgt', '2Dpred_nograd', '3Dgt']
    assert args.env in ['SD-S', 'SD-M', 'SD-L', 'RD']
    if args.env == 'RD': env = 'realball'
    elif args.env == 'SD-S': env = 'parcour_singleenv_singlecam'
    elif args.env == 'SD-M': env = 'parcour_singleenv_multicam'
    elif args.env == 'SD-L': env = 'parcour_multienv_multicam'
    assert args.lr is not None
    assert args.forec in ['exactNDE', 'analytic']
    seed = args.seed if args.seed is not None else 42
    paper_general(args.sin, args.forec,  args.lossmode, env, args.lr, args.seed)