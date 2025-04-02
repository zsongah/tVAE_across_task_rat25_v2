import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional
from tqdm import tqdm
from data.data_preparation import get_batch_random, get_batch_ss, get_batch, process_trials,segment_all
from models.stVAE import VAE, initialize_weights
from data.dataset import Dataset
import pdb
from runners.runner_utils import attach_utils_to_runner

class stVAE_runner:

    def __init__(self, config: dict, dataset: Dataset, experiment_name: str):
        super(stVAE_runner, self).__init__() # 允许子类调用父类的方法
        self.config = config
        self.data = dataset
        self.device = dataset.device
        self.beta = config.TRAIN.BETA # 用于控制 KL 散度的权重 
        self.experiment = experiment_name
        self.model = VAE(config, self.device, self.data.in_neuron_num,
                         self.data.out_neuron_num).to(self.device) # 将模型参数都移动到指定设备上。
        self.optimizer = AdamW(
            list(filter(lambda p: p.requires_grad, self.model.parameters())), #返回模型中可学习的参数
            lr=config.TRAIN.LR.INIT, # initial learnig rate + scheduler strategy
            weight_decay=config.TRAIN.WEIGHT_DECAY, # regularization, avoid overfitting
        )

        def lr_lambda(current_step): # scheduler.step() to move next step.
            if current_step < config.TRAIN.LR.WARMUP:
                return float(current_step) / float(max(1, config.TRAIN.LR.WARMUP))
            progress = float(current_step - config.TRAIN.LR.WARMUP) / float(max(1, config.TRAIN.NUM_UPDATES - config.TRAIN.LR.WARMUP))
            return 0.5 * (1. + np.cos(np.pi * progress))  # Cosine annealing
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        attach_utils_to_runner(self)

    def run(self):
        # results directories
        if not os.path.exists(self.config.RESULT_DIR): # result
            os.makedirs(self.config.RESULT_DIR)
        model_result_dir = os.path.join(self.config.RESULT_DIR, self.model.model_type)
        if not os.path.exists(model_result_dir):
            os.makedirs(model_result_dir)
        # figs directories
        if not os.path.exists(self.config.FIG_DIR): # figure
            os.makedirs(self.config.FIG_DIR)
        model_fig_dir = os.path.join(self.config.FIG_DIR, self.model.model_type)
        if not os.path.exists(model_fig_dir):
            os.makedirs(model_fig_dir)
        # runs directories and logger
        self.experiment_dir = self._create_experiment_dir()
        self.logger = self._setup_logger()

        #将 initialize_weights 函数应用于 self.model 的所有子模块。  
        self.model.apply(initialize_weights) 
        train_loss_all = []
        val_loss_all = []
        start_epoch = 0 # 用于checkpoint
        save_dict = None # 储存最终结果
        
        # tqdm 用于提供进度条
        for epoch in tqdm(range(start_epoch + 1, self.config.TRAIN.NUM_UPDATES + 1), desc='Training', unit='epoch'): 
            
            train_loss = self.train_1_epoch(epoch)  # 使用标准训练数据
            train_loss_all.extend(train_loss)
            self.scheduler.step() # 更新学习率调度器额状态
        
            if epoch > 30: # 只有variational 并且 epoch大于30 才逐渐变大
                self.beta = self.config.TRAIN.BETA * ((epoch - 10) % 20)

            if epoch % self.config.TRAIN.VAL_INTERVAL == 0: 
                results = self.evaluate()
                val_loss_all.append(results["loss"])
                self.logger.info(f"Epoch {epoch}, VALID")
                self.logger.info(f"loss: {results['loss'][0]:.5f}")
                self.logger.info(f"mse: {results['loss'][1]:.5f}")
                self.logger.info(f"kld: {results['loss'][2]:.5f}")

            save_dict = {
                'epoch': epoch,
                'config': self.config,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss_all': train_loss_all,
                'val_loss_all': val_loss_all,
            }

        print(f'| End of training | train loss {train_loss_all[-1][0]:5.5f} | valid loss {val_loss_all[-1][0]:5.5f}')

        torch.save(save_dict, f'{self.experiment_dir}/{self.experiment}.pth') # save the model after training
        
        ########## plot 1.loss, 2.latent, 3.neural data reconstruction
        results = self.evaluate()
        results["train_loss_all"] = train_loss_all # training loss
        results["val_loss_all"] = val_loss_all # val loss during training
        # self.plot_result(results, self.experiment_dir, self.experiment, self.config)
        self._plot_result(results, loss_types = ["total loss", "MSE", "KLD"])
        self.logger.info(f'Training completed | train loss {train_loss_all[-1][0]:5.5f} | valid loss {val_loss_all[-1][0]:5.5f}')

    def train_1_epoch(self, epoch = 1) -> list:
        batch_size = self.config.TRAIN.BATCH_SIZE
        train_in = self.data.train_in
        train_out = self.data.train_out
        train_trial_no = self.data.train_trial_no
        movements = self.data.train_movements
        actions = self.data.train_actions

        self.model.train()  # turn on train mode
        total_loss = [0., 0., 0]
        loss_log = []
        # start_time = time.time()

        segment_num = train_in.size(1)
        num_batches = segment_num // batch_size
        log_interval = num_batches // self.config.TRAIN.LOGS_PER_EPOCH # 几个batch记录一次
        indices = torch.randperm(segment_num).tolist()
        for batch, start_idx in enumerate(range(0, segment_num, batch_size)):
            data, _ = get_batch_random(train_in, batch_size, indices, start_idx) # data 为前
            _, targets = get_batch_random(train_out, batch_size, indices, start_idx)
            _, movement = get_batch_random(movements, batch_size, indices, start_idx)
            _, action = get_batch_random(actions, batch_size, indices, start_idx)
            _, trial_no = get_batch_random(train_trial_no, batch_size, indices, start_idx)
 
            output, mu, log_var = self.model(data)
            output_flat = output.permute(1, 0, 2).reshape(-1, self.data.out_neuron_num) # sequence + sequence
            # if is action: classication, if is movement: regression
            loss, mse, kld = self.model.loss_function(output_flat, targets, mu, log_var, self.beta)
            
            self.optimizer.zero_grad()
            loss.backward()

            # 限制模型参数的梯度值在一定范围内,以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAIN.CLIP_GRAD_NORM)
            self.optimizer.step()

            for i, item in enumerate([loss, mse, kld]):
                total_loss[i] += item.item()

            if batch % log_interval == 0 and batch > 0:
                # lr = self.optimizer.param_groups[0]['lr']
                # ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                if len(loss_log) == 0:
                    # cur_loss = total_loss[0] / (log_interval + 1)
                    loss_log.append([total_loss[i] / (log_interval + 1) for i in range(3)])
                else:
                    # cur_loss = total_loss[0] / log_interval
                    loss_log.append([total_loss[i] / log_interval for i in range(3)])

                # print(f'| epoch {epoch:3d} | {batch:2d}/{num_batches:2d} batches | '
                #       f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                #       f'beta {self.beta:.3f} | loss {cur_loss:5.5f}')
                total_loss = [0., 0., 0]
                # start_time = time.time()
        # 记录训练损失
        self.logger.info(f"Epoch {epoch}, TRAIN")
        self.logger.info(f"loss: {loss_log[-1][0]:.5f}")
        self.logger.info(f"mse: {loss_log[-1][1]:.5f}")
        self.logger.info(f"kld: {loss_log[-1][2]:.5f}")

        return loss_log # 返回每个 batch的平均loss 

    def evaluate(self) -> dict:
        latent_dim = self.config.MODEL.LATENT_DIM
        batch_size = self.config.TRAIN.BATCH_SIZE_TEST # test batch size
        step_size = self.config.TRAIN.STEP_SIZE_TEST # test step size 
        self.model.eval()  # turn on evaluation mode
        total_loss = [0., 0., 0] # 
        predictions = np.empty((0, self.data.out_neuron_num)) # output
        truth = np.empty((0, self.data.out_neuron_num))
        # if self.data.task == '2MC':
        movements = np.empty((0, 2))
        actions = np.empty((0, 1))
        # else:
        #     movements = np.empty((0, 1))
        trials = np.empty((0, 1), dtype=int)
        events = np.empty((0, 1), dtype=int)
        latent_mu = np.empty((0, latent_dim))
        latent_std = np.empty((0, latent_dim))
        with torch.no_grad(): # 禁止自动求导
            for start_idx in range(0, self.data.test_in.size(1), batch_size):
                data, _, _ = get_batch_ss(self.data.test_in, batch_size, start_idx, step_size)
                _, _, targets = get_batch_ss(self.data.test_out, batch_size, start_idx, step_size)
                _, _, movement = get_batch_ss(self.data.test_movements, batch_size, start_idx, step_size) # previously from input: eval_events 
                _, _, trial_no = get_batch_ss(self.data.test_trial_no, batch_size, start_idx, step_size)# previously from input: eval_trial_no 
                _, _, event = get_batch_ss(self.data.test_events, batch_size, start_idx, step_size)
                _, _, action = get_batch_ss(self.data.test_actions, batch_size, start_idx, step_size)
              
                output, mu, log_var = self.model(data)
                output_valid = output[-step_size:].permute(1, 0, 2).reshape(-1, self.data.out_neuron_num)
                mu_valid = mu[-step_size:]
                log_var_valid = log_var[-step_size:]
                # 取后seq-1个数据做evaluation
                # _, targets, _ = get_batch_ss(self.data.test_out, batch_size, start_idx, step_size)
                # _, events, _ = get_batch_ss(eval_events, batch_size, start_idx, step_size)
                # _, trial_no, _ = get_batch_ss(eval_trial_no, batch_size, start_idx, step_size)
                # output, mu, log_var, predicted_behav = self.model(data)
                # output_valid = output.permute(1, 0, 2).reshape(-1, self.data.out_neuron_num)
                # mu_valid = mu
                # log_var_valid = log_var

                loss, mse, kld = self.model.loss_function(output_valid, targets, mu_valid, log_var_valid,
                                                                          self.beta)

                for i, item in enumerate([loss, mse, kld]):
                    total_loss[i] += output_valid.size(0) * item.item()
                predictions = np.vstack((predictions, output_valid.cpu().numpy())) # spike_count
                truth = np.vstack((truth, targets.cpu().numpy()))  # spike_count
                movements = np.vstack((movements, movement.cpu().numpy())) # real trajectory
                trials = np.vstack((trials, trial_no.cpu().numpy())) # trial_no
                events = np.vstack((events, event.cpu().numpy())) # event
                actions = np.vstack((actions, action.cpu().numpy())) # action

                latent_mu = np.vstack(
                    (latent_mu, mu_valid.permute(1, 0, 2).reshape(-1, latent_dim).cpu().numpy()))
                latent_std = np.vstack(
                    (latent_std, torch.exp(0.5 * log_var_valid.permute(1, 0, 2).reshape(-1, latent_dim)).cpu().numpy()))

        loss = [total_loss[i] / len(predictions) for i in range(3)] # 每个点的loss
        results = {'loss': loss,
                   'truth': truth,
                   'predictions': predictions,
                   'movements': movements,
                   'actions': actions,
                   'trials': trials,
                   'events': events,
                   'latent_mu': latent_mu,
                   'latent_std': latent_std,
                   }

        return results

    def eval_train(self) -> list:
        data_segment = segment_all(self.data.data_before_segment,self.config.MODEL.TIME_WINDOW,
                                        self.config.TRAIN.STEP_SIZE_TEST, self.config.TRAIN.STEP_SIZE_TEST)
        train_in = data_segment['M1_train'].to(self.device)
        train_out = data_segment['M1_train'].to(self.device)
        train_trial_no = data_segment['trial_No_train'].to(self.device)
        train_movements = data_segment['movements_train'].to(self.device)
        train_actions = data_segment['actions_train'].to(self.device)
        train_events = data_segment['events_train'].to(self.device)
        latent_dim = self.config.MODEL.LATENT_DIM
        batch_size = self.config.TRAIN.BATCH_SIZE_TEST # test batch size
        step_size = self.config.TRAIN.STEP_SIZE_TEST# test step size
        self.model.eval() # turn on evaluation mode
        point_num = 0
        total_loss = [0., 0., 0] # 
        predictions = np.empty((0, self.data.out_neuron_num))
        truth = np.empty((0, self.data.out_neuron_num))
        movements = np.empty((0, 2))
        actions = np.empty((0, 1))
        trials = np.empty((0, 1), dtype=int)
        events = np.empty((0, 1), dtype=int)
        latent_mu = np.empty((0, latent_dim))
        latent_std = np.empty((0, latent_dim))
        with torch.no_grad(): # 禁止自动求导
            for start_idx in range(0, self.data.train_in.size(1), batch_size):
                # 取后seq-1个数据做evaluation
                data, _, _ = get_batch_ss(train_in, batch_size, start_idx, step_size)             
                _, _, targets = get_batch_ss(train_out, batch_size, start_idx, step_size)
                _, _, movement = get_batch_ss(train_movements, batch_size, start_idx, step_size)
                _, _, trial_no = get_batch_ss(train_trial_no, batch_size, start_idx, step_size)
                _, _, event = get_batch_ss(train_events, batch_size, start_idx, step_size)
                _, _, action = get_batch_ss(train_actions, batch_size, start_idx, step_size)

                output, mu, log_var = self.model(data)
                output_valid = output[-step_size:].permute(1, 0, 2).reshape(-1, self.data.out_neuron_num)
                mu_valid = mu[-step_size:]
                log_var_valid = log_var[-step_size:]

                loss, mse, kld = self.model.loss_function(output_valid, targets, mu_valid, log_var_valid, 
                                                                            self.beta)

                for i, item in enumerate([loss, mse, kld]):
                    total_loss[i] += output_valid.size(0) * item.item()
                point_num += output_valid.size(0)

                predictions = np.vstack((predictions, output_valid.cpu().numpy()))
                truth = np.vstack((truth, targets.cpu().numpy()))
                movements = np.vstack((movements, movement.cpu().numpy()))
                trials = np.vstack((trials, trial_no.cpu().numpy()))
                events = np.vstack((events, event.cpu().numpy()))
                actions = np.vstack((actions, action.cpu().numpy()))

                latent_mu = np.vstack(
                    (latent_mu, mu_valid.permute(1, 0, 2).reshape(-1, latent_dim).cpu().numpy()))
                latent_std = np.vstack(
                    (latent_std, torch.exp(0.5 * log_var_valid.permute(1, 0, 2).reshape(-1, latent_dim)).cpu().numpy()))
                
        loss = [total_loss[i] / point_num for i in range(3)] # 每个点的loss
        results = {'loss': loss,
                    'truth': truth,
                    'predictions': predictions,
                    'movements': movements,
                    'actions': actions,
                    'trials': trials,
                    'events': events,
                    'latent_mu': latent_mu,
                    'latent_std': latent_std,
                    }
        return results