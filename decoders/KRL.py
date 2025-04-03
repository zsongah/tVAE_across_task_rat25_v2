import os
import sys
import numpy as np
import random
from .KRL_utils import data_proprecess
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class KRL:
    def __init__(self, model_name, latent_one_lever, latent_two_lever, trial_types_2MC, cue_start, press_lever,release_lever, seed=3):
        # 设置随机种子
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.model_name = model_name
        # data processing
        self.data_1MC, self.label_1MC = data_proprecess(latent_one_lever, cue_start, press_lever, release_lever,[1]*latent_one_lever.shape[0])
        self.data_2MC, self.label_2MC = data_proprecess(latent_two_lever, cue_start, press_lever, release_lever,trial_types_2MC)
        self.predict_label_1MC = None
        self.predict_label_2MC = None
        # KRL hyper parameters
        self.kernel_width = 2
        self.learning_rate = 0.2 # rat 25:0.2
        self.val_interval = 15 # rat 25:10
        # KRL parameters
        self.cluster_1MC = None
        self.weight_to_output_1MC = None
        self.cluster_2MC = None
        self.weight_to_output_2MC = None

    def train_on_MC1(self):
         # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            self.data_1MC, self.label_1MC, test_size=0.2, random_state=self.seed
        )
        cluster = X_train[0, :][:, np.newaxis]
        weight_to_output = np.array([[0], [0]])
        eval_acc = []
        for i_data in range(1, X_train.shape[0]):
            predict_action, chosen_probability = self._decode_krl(X_train[i_data, :][:, np.newaxis], cluster, weight_to_output)
            f_delta = self._compute_error(predict_action, y_train[i_data], chosen_probability)     
            cluster, weight_to_output = self._update_krl_params(
                X_train[i_data, :][:, np.newaxis], predict_action, f_delta, cluster, weight_to_output)
            if i_data % self.val_interval == 0:
                eval_acc.append(self.evaluate(X_val, y_val,cluster,weight_to_output))
        self.cluster_1MC = cluster
        self.weight_to_output_1MC = weight_to_output

        return eval_acc
  
    # def train_on_MC2(self):
    #     # 划分训练集和验证集
    #     X_train, X_val, y_train, y_val = train_test_split(
    #         self.data_2MC, self.label_2MC, test_size=0.2, random_state=self.seed
    #     )
    #     # 在2MC数据上测试分类器
    #     cluster = self.cluster_1MC
    #     new_row = np.zeros((1, self.weight_to_output_1MC.shape[1]))  # new action
    #     weight_to_output = np.vstack([self.weight_to_output_1MC, new_row])
    #     # weight_to_output = self.weight_to_output_1MC
    #     eval_acc = []
    #     for i_data in range(X_train.shape[0]):
    #         predict_action,chosen_probability = self._decode_krl(X_train[i_data, :][:, np.newaxis], cluster, weight_to_output)
    #         f_delta = self._compute_error(predict_action, y_train[i_data], chosen_probability)
    #         cluster, weight_to_output = self._update_krl_params(
    #             X_train[i_data, :][:, np.newaxis], predict_action, f_delta, cluster, weight_to_output)
    #         if i_data % self.val_interval == 0:
    #             eval_acc.append(self.evaluate(X_val, y_val,cluster,weight_to_output))
    #     self.cluster_2MC = cluster
    #     self.weight_to_output_2MC = weight_to_output
    #     return eval_acc
    
    def train_on_MC2(self, n_splits=5):
        """
        使用 5-fold cross-validation 对 MC2 数据进行训练和评估。
        
        参数:
            n_splits: int, 折数 (默认5)
        返回:
            all_eval_acc: list, 每个 fold 的 eval_acc 列表
        """
        # 使用 KFold 划分数据
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        all_eval_acc = []

        # 开始 5-fold 训练和评估
        for fold_index, (train_index, val_index) in enumerate(kf.split(self.data_2MC)):
            # print(f"Fold {fold_index + 1}/{n_splits}:")
            
            # 获取当前折的训练和验证数据
            X_train, X_val = self.data_2MC[train_index], self.data_2MC[val_index]
            y_train, y_val = self.label_2MC[train_index], self.label_2MC[val_index]

            # 初始化 cluster 和 weight_to_output
            cluster = self.cluster_1MC
            new_row = np.zeros((1, self.weight_to_output_1MC.shape[1]))  # 添加新 action
            weight_to_output = np.vstack([self.weight_to_output_1MC, new_row])

            # 存储当前 fold 的验证集准确率
            eval_acc = []
            
            # 训练模型
            for i_data in range(X_train.shape[0]):
                predict_action, chosen_probability = self._decode_krl(
                    X_train[i_data, :][:, np.newaxis], cluster, weight_to_output
                )
                f_delta = self._compute_error(predict_action, y_train[i_data], chosen_probability)
                cluster, weight_to_output = self._update_krl_params(
                    X_train[i_data, :][:, np.newaxis], predict_action, f_delta, cluster, weight_to_output
                )

                # 在验证集上进行评估
                if i_data % self.val_interval == 0:
                    eval_acc.append(self.evaluate(X_val, y_val, cluster, weight_to_output))

            # 保存当前 fold 的评估结果
            all_eval_acc.append(eval_acc)

        # 最后一次折的参数保存为 cluster_2MC 和 weight_to_output_2MC
        self.cluster_2MC = cluster
        self.weight_to_output_2MC = weight_to_output

        # 返回所有 fold 的 eval_acc 列表
        return np.array(all_eval_acc)

    def evaluate(self, data, label,cluster,weight_to_output):
        # 在验证集上测试分类器
        val_predictions = []
        for i_data in range(data.shape[0]):
            predict_action, _ = self._decode_krl(
                data[i_data, :][:, np.newaxis], cluster, weight_to_output
            )
            val_predictions.append(predict_action)
        # 计算验证集准确率
        val_accuracy = accuracy_score(label, val_predictions)
        return val_accuracy


    # def test(self):
    #     # 在2MC数据上测试分类器
    #     pred_act_2mc = []
    #     for i_data in range(self.data_2MC.shape[0]):
    #         predict_action,chosen_probability = self._decode_krl(self.data_2MC[i_data, :][:, np.newaxis], self.cluster, self.weight_to_output)
    #         pred_act_2mc.append(predict_action)
    #     # 计算预测准确率
    #     self.predict_label_2MC = np.array(pred_act_2mc)
    #     acc = accuracy_score(self.label_2MC, self.predict_label_2MC)
    #     improved_acc = acc - 1/3  # 相对于随机分类的提升
    #     return improved_acc
   
    # def _train_krl(self, training_data):
    #     # training_data shape: (feature_dim, num_data)
    #     # 训练KRL分类器
    #     shuffled_data = training_data[:, np.random.permutation(training_data.shape[1])]
    #     spike = shuffled_data[:-1, :]
    #     label = shuffled_data[-1, :]

    #     cluster = spike[:, 0][:, np.newaxis]
    #     weight_to_output = np.array([[0], [0]])
    #     predict = []
    #     for i_data in range(1, spike.shape[1]):
    #         predict_action, chosen_probability = self._decode_krl(spike[:, i_data][:, np.newaxis], cluster, weight_to_output)
    #         f_delta = self._compute_error(predict_action, label[i_data], chosen_probability)
            
    #         cluster, weight_to_output = self._update_krl_params(
    #             spike[:, i_data][:, np.newaxis], predict_action, f_delta, cluster, weight_to_output)
    #         predict.append(predict_action)

    #     return cluster, weight_to_output, np.array(predict)
    
    def _decode_krl(self, spike, cluster, weight_to_output):

        temp_kernel = self._gaussian_kernel(spike, cluster)
        q_output = weight_to_output.dot(temp_kernel.T)
        p_output = self._softmax(q_output)
        predict_action = np.random.choice(range(len(p_output)), p=p_output.flatten())
        chosen_probability = p_output[predict_action]
        
        return predict_action, chosen_probability

    def _update_krl(self, data):
        # RL更新
        spike = data[:-1, :]
        label = data[-1, :]
        true_cluster_label = [label[0]]
        
        for i_data in range(spike.shape[1]): 
            predict_action, chosen_probability = self._decode_krl(spike[:, i_data][:, np.newaxis])
            f_delta = self._compute_error(predict_action, label[i_data], chosen_probability)
            
            self.cluster, self.weight_to_output, self.centroid, true_cluster_label = self._update_krl_params(
                spike[:, i_data][:, np.newaxis], predict_action, f_delta, self.cluster, self.weight_to_output
            )
    
    def _compute_error(self, predict_action, true_label, chosen_probability):
        # 计算错误
        if predict_action == true_label:
            f_delta = 1 - chosen_probability
        else:
            f_delta = -1
        return f_delta
    
    def _softmax(self, x):
        # Softmax函数
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _gaussian_kernel(self, x, y):
        # 高斯核函数
        return np.exp(-np.sum((x - y)**2,axis=0) / (2 * self.kernel_width**2))
    
    def _update_krl_params(self, spike, predict_action, f_delta, cluster, weight_to_output):
        tempWeight = np.zeros((weight_to_output.shape[0],1))
        tempWeight[predict_action] = self.learning_rate * f_delta
        new_weight_to_output = np.hstack((weight_to_output, tempWeight))
        new_cluster = np.hstack((cluster, spike))
        return new_cluster, new_weight_to_output