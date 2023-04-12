# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import json
import h5py
from tqdm import tqdm, trange
from layers.summarizer import PGL_SUM
from utils import TensorboardWriter

from os import listdir
from os.path import join
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
from inference import inference


class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """ Function for constructing the PGL-SUM model of its key modules and parameters."""
        # Model creation
        self.model = PGL_SUM(input_size=self.config.input_size,
                             output_size=self.config.input_size,
                             num_segments=self.config.n_segments,
                             heads=self.config.heads,
                             fusion=self.config.fusion,
                             pos_enc=self.config.pos_enc).to(self.config.device)
        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        if self.config.mode == 'train':
            # Optimizer initialization
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)
            self.writer = TensorboardWriter(str(self.config.log_dir))
            

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    criterion = nn.MSELoss()

    def train(self):
        """ Main function to train the PGL-SUM model. """
        max_f_score = -1
        for epoch_i in tqdm(range(self.config.n_epochs), desc='Epoch', ncols=80):
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader) / self.config.batch_size)  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for _ in tqdm(range(num_batches), desc='Batch', ncols=80, leave=False):
                # ---- Training ... ----#
                if self.config.verbose:
                    tqdm.write('Time to train the model...')

                self.optimizer.zero_grad()
                for _ in tqdm(range(self.config.batch_size), desc='Video', ncols=80, leave=False):
                    frame_features, target, _, _, _, _ = next(iterator)

                    frame_features = frame_features.to(self.config.device)
                    target = target.to(self.config.device)

                    output, weights = self.model(frame_features.squeeze(0))
                    loss = self.criterion(output.squeeze(0), target.squeeze(0))

                    if self.config.verbose:
                        tqdm.write(f'[{epoch_i}] loss: {loss.item()}')

                    loss.backward()
                    loss_history.append(loss.data)
                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')

            self.writer.update_loss(loss, epoch_i, 'loss_epoch')
            # Uncomment to save parameters at checkpoint
            # if not os.path.exists(self.config.save_dir):
            #     os.makedirs(self.config.save_dir)
            # ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pt'
            # tqdm.write(f'Save parameters at {ckpt_path}')
            # torch.save(self.model.state_dict(), ckpt_path)

            # # Model data
            # model_path = f"../PGL-SUM/Summaries/PGL-SUM/exp1/{self.config.video_type}/models/split{self.config.split_index}/seed{self.config.seed}"
            # model_file = sorted([f for f in listdir(model_path)])
            # eval_metric = 'avg' if self.config.video_type.lower() == 'tvsum' else 'max'
            # # Read current split
            # split_file = f"../PGL-SUM/data/datasets/splits/{self.config.video_type.lower()}_splits.json"
            # with open(split_file) as f:
            #     data = json.loads(f.read())
            #     test_keys = data[self.config.split_index]["test_keys"]
            # # Dataset path
            # dataset_path = f"../PGL-SUM/data/datasets/{self.config.video_type}/eccv16_dataset_{self.config.video_type.lower()}_google_pool5.h5"
            # trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
            #                     fusion="add", pos_enc="absolute")
            # trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))
            # # os.remove(join(model_path, model_file[-1]))
            # # f_score = inference(trained_model, dataset_path, test_keys, eval_metric, self.writer, epoch_i)
            f_score = self.evaluate(epoch_i)

            if not os.path.exists(self.config.save_dir):
                os.makedirs(self.config.save_dir)
            model_path = f"../PGL-SUM/Summaries/PGL-SUM/exp1/{self.config.video_type}/models/split{self.config.split_index}/seed{self.config.seed}"
            model_file = sorted([f for f in listdir(model_path)])
            if (max_f_score < f_score):
                max_f_score = f_score
                if len(model_file) > 0:
                    os.remove(join(model_path, model_file[-1]))
                ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pt'
                tqdm.write(f'Save parameters at {ckpt_path}')
                torch.save(self.model.state_dict(), ckpt_path)

        # Print results of of val best-F scores
        if not os.path.exists(self.config.f_score_dir):
            os.makedirs(self.config.f_score_dir)
        with open (self.config.f_score_dir.joinpath('max_f_score.txt'), 'a') as file:
            file.writelines("seed" + str(self.config.seed) + ': ' + str(max_f_score) + '\n')

        ### Inference part
        test_f_score_dict = {}
        eval_metric = 'avg' if self.config.video_type.lower() == 'tvsum' else 'max'
        # Read current split
        ### TODO
        # Change this to 622splits later on
        split_file = f"../PGL-SUM/data/datasets/splits/{self.config.video_type.lower()}_val_splits.json"
        with open(split_file) as f:
            data = json.loads(f.read())
            test_keys = data[self.config.split_index]["test_keys"]
        # Dataset path
        dataset_path = f"../PGL-SUM/data/datasets/{self.config.video_type}/eccv16_dataset_{self.config.video_type.lower()}_google_pool5.h5"
        trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute")
        trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))
        test_f_score = inference(trained_model, dataset_path, test_keys, eval_metric)
        if not os.path.exists(self.config.test_f_score_dir):
            os.makedirs(self.config.test_f_score_dir)
        with open (self.config.test_f_score_dir.joinpath('test_f_score.txt'), 'a') as file:
            file.writelines("seed" + str(self.config.seed) + ': ' + str(test_f_score) + '\n')
        
            


    def evaluate(self, epoch_i, save_weights=False):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        :param bool save_weights: Optionally, the user can choose to save the attention weights in a (large) h5 file.
        """
        self.model.eval()

        weights_save_path = self.config.score_dir.joinpath("weights.h5")
        out_scores_dict = {}
        video_fscores = []
        eval_metric = 'avg' if self.config.video_type.lower() == 'tvsum' else 'max'
        for frame_features, video_name, user_summary, sb, n_frames, positions in tqdm(self.val_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, input_size]
            frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
            with torch.no_grad():
                scores, attn_weights = self.model(frame_features)  # [1, seq_len]
                scores = scores.squeeze(0).cpu().numpy().tolist()
                attn_weights = attn_weights.cpu().numpy()
                summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
                f_score = evaluate_summary(summary, user_summary, eval_metric)
                out_scores_dict[video_name] = scores
                video_fscores.append(f_score)
            if not os.path.exists(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            scores_save_path = self.config.score_dir.joinpath(f"{self.config.video_type}_{epoch_i}.json")
            with open(scores_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(scores_save_path)}.')
                json.dump(out_scores_dict, f)
            scores_save_path.chmod(0o777)

            if save_weights:
                with h5py.File(weights_save_path, 'a') as weights:
                    weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)
        self.writer.update_loss(np.mean(video_fscores), epoch_i, 'Val F-score_epoch')
        return np.mean(video_fscores)


if __name__ == '__main__':
    pass
