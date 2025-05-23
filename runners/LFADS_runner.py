import torch
import os
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from models.LFADS import LFADS
from data.dataset import Dataset
from torch.optim.lr_scheduler import LambdaLR
from data.data_preparation import get_batch_random, get_batch_ss, get_batch, process_trials, segment_all
from matplotlib import pyplot as plt
from runners.runner_utils import attach_utils_to_runner

class LFADS_runner:

    def __init__(self, config:dict, dataset:Dataset, experiment_name:str):
        self.config = config
        self.data = dataset
        self.experiment = experiment_name
        self.device = dataset.device
        self.model = LFADS(config, self.device, 'LFADS').to(self.device)
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
        # model_result_dir = os.path.join(self.config.RESULT_DIR, self.model.model_type)
        # if not os.path.exists(model_result_dir):
        #     os.makedirs(model_result_dir)
        # model_fig_dir = os.path.join(self.config.FIG_DIR, self.model.model_type)
        # if not os.path.exists(model_fig_dir):
        #     os.makedirs(model_fig_dir)

        # runs directories and logger
        self.experiment_dir = self._create_experiment_dir()
        self.logger = self._setup_logger()

        train_loss_all = []
        val_loss_all = []
        start_epoch = 0 
        save_dict = None 

        for epoch in tqdm(range(start_epoch+1,self.config.TRAIN.NUM_UPDATES+1), desc='Training',unit='epoch'):
            self.model.current_epoch = epoch
            train_loss = self.train_1_epoch(epoch)
            train_loss_all.extend(train_loss)
            self.scheduler.step()

            if epoch % self.config.TRAIN.VAL_INTERVAL == 0:
                results = self.evaluate()
                val_loss_all.append(results['loss'])
                self.logger.info(f"Epoch {epoch}, VALID")
                self.logger.info(f"total_loss: {results['loss'][0]:.5f}")
                self.logger.info(f"recon_loss: {results['loss'][1]:.5f}")
                self.logger.info(f"l2_loss: {results['loss'][2]:.5f}")
                self.logger.info(f"kl_loss: {results['loss'][3]:.5f}")

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
        torch.save(save_dict,f'{self.experiment_dir}/{self.experiment}.pth')

        results = self.evaluate()
        results["train_loss_all"] = train_loss_all # training loss
        results["val_loss_all"] = val_loss_all # val loss during training
        # self.plot_result(results, model_fig_dir, self.experiment, self.config)
        # self.plot_result(results, self.experiment_dir, self.experiment, self.config)
        self._plot_result(results, loss_types = ['total_all', 'recon_loss', 'l2_loss', 'kl_loss'])
        self.logger.info(f'Training completed | train loss {train_loss_all[-1][0]:5.5f} | valid loss {val_loss_all[-1][0]:5.5f}')
    
    def train_1_epoch(self, epoch) -> list:
        batch_size = self.config.TRAIN.BATCH_SIZE
        train_in = self.data.train_in # [len,batch,neuron]
        train_out = self.data.train_out
        train_trial_no = self.data.train_trial_no

        self.model.train()
        total_loss = [0., 0., 0., 0.]
        loss_log = []

        segment_num = train_in.size(1)
        num_batches = segment_num //batch_size
        log_interval = num_batches // self.config.TRAIN.LOGS_PER_EPOCH

        indices = torch.randperm(segment_num).tolist()
        for batch_idx, start_idx in enumerate(range(0, segment_num, batch_size)):
            data, _ = get_batch_random(train_in, batch_size, indices, start_idx) # data 为前
            _, targets = get_batch_random(train_out, batch_size, indices, start_idx)

            # batch_size, seq_len, input_dim
            data = data.permute(1, 0, 2)
    
            total_all, recon_loss, l2_loss, kl_loss = self.model.loss_function(data, targets) 

            self.optimizer.zero_grad()
            total_all.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAIN.CLIP_GRAD_NORM)
            self.optimizer.step()

            for i, item in enumerate([total_all, recon_loss, l2_loss, kl_loss]):
                total_loss[i] += item.item()

            if batch_idx % log_interval == 0 and batch_idx > 0:
                if len(loss_log) == 0:
                    loss_log.append([total_loss[i] / (log_interval + 1) for i in range(4)])
                else:
                    loss_log.append([total_loss[i] / log_interval for i in range(4)])
                total_loss = [0., 0., 0., 0.]
        # 记录训练损失
        self.logger.info(f"Epoch {epoch}, TRAIN")
        self.logger.info(f"total_loss: {loss_log[-1][0]:.5f}")
        self.logger.info(f"recon_loss: {loss_log[-1][1]:.5f}")
        self.logger.info(f"l2_loss: {loss_log[-1][2]:.5f}")
        self.logger.info(f"kl_loss: {loss_log[-1][3]:.5f}")

        return loss_log

    def evaluate(self)->dict:
        latent_dim = self.config.MODEL.LATENT_DIM
        batch_size = self.config.TRAIN.BATCH_SIZE_TEST
        step_size = self.config.TRAIN.STEP_SIZE_TEST
        self.model.eval()
        total_loss = [0., 0., 0., 0.]
        prediction = np.empty((0,self.config.MODEL.ENCOD_DATA_DIM))
        truth = np.empty((0,self.config.MODEL.ENCOD_DATA_DIM))
        movements = np.empty((0, 2))
        actions = np.empty((0, 1))
        trials = np.empty((0, 1), dtype=int)
        events = np.empty((0, 1), dtype=int)
        latent_mu = np.empty((0, latent_dim))
        predictions = np.empty((0, self.config.MODEL.ENCOD_DATA_DIM)) # output

        with torch.no_grad():
            for start_idx in range(0, self.data.test_in.size(1), batch_size):
                data, _, _ = get_batch_ss(self.data.test_in, batch_size, start_idx, step_size)
                _, _, targets = get_batch_ss(self.data.test_out, batch_size, start_idx, step_size)
                _, _, movement = get_batch_ss(self.data.test_movements, batch_size, start_idx, step_size) # previously from input: eval_events 
                _, _, trial_no = get_batch_ss(self.data.test_trial_no, batch_size, start_idx, step_size)# previously from input: eval_trial_no 
                _, _, event = get_batch_ss(self.data.test_events, batch_size, start_idx, step_size)
                _, _, action = get_batch_ss(self.data.test_actions, batch_size, start_idx, step_size)
                
                # batch_size, seq_len, input_dim
                data = data.permute(1, 0, 2)
                loss_all, recon_loss, l2_loss, kl_loss = self.model.loss_function(data, targets)
                output = self.model(data) 

                pred_valid = output['pred'][:,-step_size:,:].reshape(-1, self.config.MODEL.ENCOD_DATA_DIM)
                latent_valid = output['factors'][:,-step_size:,:].reshape(-1, self.config.MODEL.LATENT_DIM)


                for i, item in enumerate([loss_all, recon_loss, l2_loss, kl_loss]):
                    total_loss[i] += item.item()
                predictions = np.vstack((predictions, pred_valid.cpu().numpy())) # spike_count
                truth = np.vstack((truth, targets.cpu().numpy()))  # spike_count
                movements = np.vstack((movements, movement.cpu().numpy())) # real trajectory
                trials = np.vstack((trials, trial_no.cpu().numpy())) # trial_no
                events = np.vstack((events, event.cpu().numpy())) # event
                actions = np.vstack((actions, action.cpu().numpy())) # action
                latent_mu = np.vstack((latent_mu, latent_valid.cpu().numpy()))

        loss = total_loss   
        results = {'loss': loss,
                   'truth': truth,
                   'predictions': predictions,
                   'movements': movements,
                   'actions': actions,
                   'trials': trials,
                   'events': events,
                   'latent_mu': latent_mu,
                   }
        return results

    def eval_train(self) -> dict:
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

        total_loss = [0., 0., 0.,0] # 
        predictions = np.empty((0, self.data.out_neuron_num))
        truth = np.empty((0, self.data.out_neuron_num))
        movements = np.empty((0, 2))
        actions = np.empty((0, 1))
        trials = np.empty((0, 1), dtype=int)
        events = np.empty((0, 1), dtype=int)
        latent_mu = np.empty((0, latent_dim))

        with torch.no_grad(): # 禁止自动求导
            for start_idx in range(0, self.data.train_in.size(1), batch_size):
                # 取后seq-1个数据做evaluation
                data, _, _ = get_batch_ss(train_in, batch_size, start_idx, step_size)             
                _, _, targets = get_batch_ss(train_out, batch_size, start_idx, step_size)
                _, _, movement = get_batch_ss(train_movements, batch_size, start_idx, step_size)
                _, _, trial_no = get_batch_ss(train_trial_no, batch_size, start_idx, step_size)
                _, _, event = get_batch_ss(train_events, batch_size, start_idx, step_size)
                _, _, action = get_batch_ss(train_actions, batch_size, start_idx, step_size)

                # batch_size, seq_len, input_dim
                data = data.permute(1, 0, 2)
                loss_all, recon_loss, l2_loss, kl_loss = self.model.loss_function(data, targets)
                output = self.model(data) 

                pred_valid = output['pred'][:,-step_size:,:].reshape(-1, self.config.MODEL.ENCOD_DATA_DIM)
                latent_valid = output['factors'][:,-step_size:,:].reshape(-1, self.config.MODEL.LATENT_DIM)

                for i, item in enumerate([loss_all, recon_loss, l2_loss, kl_loss]):
                    total_loss[i] += item.item()
                predictions = np.vstack((predictions, pred_valid.cpu().numpy())) # spike_count
                truth = np.vstack((truth, targets.cpu().numpy()))  # spike_count
                movements = np.vstack((movements, movement.cpu().numpy())) # real trajectory
                trials = np.vstack((trials, trial_no.cpu().numpy())) # trial_no
                events = np.vstack((events, event.cpu().numpy())) # event
                actions = np.vstack((actions, action.cpu().numpy())) # action
                latent_mu = np.vstack((latent_mu, latent_valid.cpu().numpy()))

        loss = total_loss   
        results = {'loss': loss,
                   'truth': truth,
                   'predictions': predictions,
                   'movements': movements,
                   'actions': actions,
                   'trials': trials,
                   'events': events,
                   'latent_mu': latent_mu,
                   }
        return results

    # @staticmethod # 静态方法,不需要实例化就可以调用
    # def plot_result(results, save_dir, target_file, config):
    #     # print(f"Saving plot to: {save_dir}/{target_file}_{epoch}.png")
    #     # inferred smoothed spike counts
    #     plot_time = np.arange(0, 10, 0.1) # 1 represent 0.1 s
    #     plot_indexes = range(0, 100)
    #     fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    #     for i in range(4):
    #         for j in range(4):
    #             # plot actions
    #             # colors = action_colors[results["movements"][plot_indexes, 0].astype(int)]
    #             # axs[i, j].bar(plot_time, np.ones(len(plot_indexes)), color=colors, bottom=0, alpha=0.5)
    #             # plot firing rate of neuron i
    #             # truth_smooth = gaussian_filter1d(results["truth"][plot_indexes, i * 4 + j], sigma=10)
    #             # predictions_smooth = gaussian_filter1d(results["predictions"][plot_indexes, i * 4 + j], sigma=10)
    #             # No smoothing for the results
    #             truth = results["truth"][plot_indexes, i * 4 + j]
    #             predictions =results["predictions"][plot_indexes, i * 4 + j]
    #             axs[i, j].plot(plot_time, truth, label='Ground Truth', color='k')
    #             axs[i, j].plot(plot_time, predictions, label='Predictions', color='g')
    #             axs[i, j].set_ylabel('Smoothed spike counts')
    #             axs[i, j].set_xlabel('Times (s)')
    #             axs[i, j].set_title(f'Neuron {i * 4 + j + 1}', fontsize=12)
    #             # plot spike trains of neuron i
    #             # spike_times = plot_time[results["truth"][plot_indexes, i * 2 + j] == 1]
    #             # axs[i, j].vlines(spike_times, 0.75, 0.95, color='k', linewidth=0.3, label='Truth Spikes')
    #             # spike_times = plot_time[results["predicted_spikes"][plot_indexes, i * 2 + j] == 1]
    #             # axs[i, j].vlines(spike_times, 0.5, 0.7, color='g', linewidth=0.3, label='Predicted Spikes')
    #             # axs[i, j].set_title('Neuron ' + str(i * 2 + j + 1))
    #             # axs[i, j].set_xticklabels([])
    #     handles, labels = axs[0, 0].get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=20)
    #     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.95, hspace=0.4, wspace=0.3)
    #     plt.savefig(f'{save_dir}/{target_file}_spike.png', bbox_inches='tight',dpi=300)

    #     # inferred latent variables
    #     fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    #     for d in range(results["latent_mu"].shape[1]):
    #         row = d // 4  # 计算行索引
    #         col = d % 4   # 计算列索引
    #         if d < 16:
    #             axs[row,col].plot(plot_time, results["latent_mu"][plot_indexes, d])
    #             #axs[3, 1].plot(plot_time, gaussian_filter1d(results["latent_mu"][plot_indexes, d], sigma=10))
    #     plt.savefig(f'{save_dir}/{target_file}_latent.png', bbox_inches='tight')

    #     # plot loss
    #     fig, axs = plt.subplots(4,figsize=(15,10))
    #     loss_type = ["total loss", "recon_loss", "l2_loss", "kl_loss"]
    #     for i in range(0, 4):
    #         axs[i].plot(np.arange(1, len(results['train_loss_all'])+1)/config.TRAIN.LOGS_PER_EPOCH,
    #                     np.array(results['train_loss_all'])[:, i], label='train loss')
    #         axs[i].plot(np.arange(config.TRAIN.VAL_INTERVAL, config.TRAIN.NUM_UPDATES+1,
    #                               config.TRAIN.VAL_INTERVAL),
    #                     np.array(results['val_loss_all'])[:, i], label='test loss')
    #         axs[i].set_ylabel(loss_type[i])
    #     axs[2].set_xlabel('epoch')
    #     axs[2].legend(loc="upper right")

    #     plt.savefig(f'{save_dir}/{target_file}_learning_curve.png', bbox_inches='tight')

