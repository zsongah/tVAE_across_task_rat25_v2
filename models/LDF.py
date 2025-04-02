import numpy as np
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
import pickle
import os


class LDF:
    def __init__(self, config, model_type = 'LDF'):
        super(LDF, self).__init__()

        self.config = config
        self.model_type = model_type
        self.obs_dim = config.MODEL.DIM_Y
        self.latent_dim = config.MODEL.LATENT_DIM

        self.kf = None
        self.scaler = StandardScaler()

    def __call__(self, y):
        # -------- 数据标准化 --------
        y_scaled = self.scaler.transform(y)

        # -------- 状态估计 --------
        state_means, _ = self.kf.filter(y_scaled)

        # -------- 计算预测的观测值（标准化后） --------
        y_pred_scaled = state_means @ self.kf.observation_matrices.T

        # -------- 反标准化预测的观测值 --------
        y_pred = self.scaler.inverse_transform(y_pred_scaled)

        return y_pred, state_means

    def fit(self, y, para_init, n_init=1, n_iter=5):
        
        y_scaled = self.scaler.fit_transform(y)
        best_log_likelihood = -np.inf
        best_kf = None

        for _ in range(n_init):
            try:
                kf = KalmanFilter(
                    transition_matrices=para_init['A'],
                    observation_matrices=para_init['C'],
                    transition_covariance=para_init['Q'],
                    observation_covariance=para_init['R'],
                    initial_state_mean=np.zeros(self.latent_dim),
                    initial_state_covariance=np.eye(self.latent_dim)
                )
                kf = kf.em(y_scaled, n_iter=n_iter)

                score = kf.loglikelihood(y_scaled).sum()
                if score > best_log_likelihood:
                    best_log_likelihood = score
                    best_kf = kf
            except Exception as e:
                print(f"Kalman EM failed: {e}")
                continue

        if best_kf is None:
            raise RuntimeError("Kalman Filter training failed.")

        self.kf = best_kf

    def save_checkpoint(self, path):
        if self.kf is None:
            raise ValueError("Model has not been trained.")
        
        checkpoint = {
            "kf_params": {
                "transition_matrices": self.kf.transition_matrices,
                "observation_matrices": self.kf.observation_matrices,
                "transition_covariance": self.kf.transition_covariance,
                "observation_covariance": self.kf.observation_covariance,
                "initial_state_mean": self.kf.initial_state_mean,
                "initial_state_covariance": self.kf.initial_state_covariance
            },
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.kf = KalmanFilter(
            transition_matrices=checkpoint['kf_params']['transition_matrices'],
            observation_matrices=checkpoint['kf_params']['observation_matrices'],
            transition_covariance=checkpoint['kf_params']['transition_covariance'],
            observation_covariance=checkpoint['kf_params']['observation_covariance'],
            initial_state_mean=checkpoint['kf_params']['initial_state_mean'],
            initial_state_covariance=checkpoint['kf_params']['initial_state_covariance']
        )

        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]
        print(f"Checkpoint loaded from {path}")