# -*- coding: utf-8 -*-
from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config(mode='train')
    val_config = get_config(mode='val')
    test_config = get_config(mode='test')

    print(config)
    print(val_config)
    print(test_config)
    print('Currently selected split_index:', config.split_index)
    train_loader = get_loader(config.mode, config.video_type, config.split_index)
    val_loader = get_loader(val_config.mode, val_config.video_type, val_config.split_index)
    test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)
    solver = Solver(config, train_loader, val_loader, test_loader)

    solver.build()
    solver.evaluate(-1)	 # evaluates the summaries using the initial random weights of the network
    solver.train()

# Validation git push demo
# tensorboard --logdir '../PGL-SUM/Summaries/PGL-SUM/'
