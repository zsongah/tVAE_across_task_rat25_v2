from models.LDF import LDF
from data.dataset import Dataset
from runners.runner_utils import create_experiment_dir
import numpy as np


class LDF_runner:
    def __init__(self, config:dict, dataset:Dataset, experiment_name:str):

        self.config = config
        self.data = dataset
        self.experiment = experiment_name
        self.model = LDF(config, 'LDF')
        self.latent_dim = config.MODEL.LATENT_DIM
        self.y_dim = config.MODEL.DIM_Y
        self.n_init = config.TRAIN.n_init
        self.n_iter = config.TRAIN.n_iter

    def run(self):
        self.experiment_dir = create_experiment_dir(self)
        y_train = self.data.data_before_segment['M1_train'].numpy().transpose()

        # 估计 LDS 参数 from MC1
        para_init = { 
            'A': np.eye(self.latent_dim) + 0.01 * np.random.randn(self.latent_dim, self.latent_dim),
            'C': np.random.randn(self.y_dim, self.latent_dim),
            'Q': np.eye(self.latent_dim) * 0.1,
            'R': np.eye(self.y_dim) * 0.1
        }
        self.model.fit(y_train, para_init, n_iter=self.n_init, n_init=self.n_iter)

        self.model.save_checkpoint(f'{self.experiment_dir}/{self.experiment}.pkl')
           
    def evaluate(self):
        y_test = self.data.data_before_segment['M1_test'].numpy().transpose()
        y_pred_test, latent_test = self.model(y_test)
        results = {
            "predictions": y_pred_test,
            "latent_mu": latent_test,
            "truth": y_test,
            "events": self.data.data_before_segment['events_test'].numpy().transpose(),
            "trials": self.data.data_before_segment['trial_No_test'].numpy().transpose(),
            "actions": self.data.data_before_segment['actions_test'].numpy().transpose()
        }
        return results
    
    def eval_train(self):
        y_train = self.data.data_before_segment['M1_train'].numpy().transpose()
        y_pred_train, latent_train = self.model(y_train)
        results = {
            "predictions": y_pred_train,
            "latent_mu": latent_train,
            "truth": y_train,
            "events": self.data.data_before_segment['events_train'].numpy().transpose(),
            "trials": self.data.data_before_segment['trial_No_train'].numpy().transpose(),
            "actions": self.data.data_before_segment['actions_train'].numpy().transpose()
        }
        return results
    
    def load_model(self, path):
        self.model.load_checkpoint(path)