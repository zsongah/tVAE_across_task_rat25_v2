# utils/runner_utils.py
import os
import logging
import datetime
import yaml
import numpy as np
import matplotlib.pyplot as plt

# 注意：函数名去掉了下划线前缀，以便与attach_utils_to_runner中的引用匹配
def setup_logger(self, logger_name=None):
    """设置日志系统，只记录到文件，不输出到控制台"""
    # 如果未提供logger_name，使用默认名称
    if logger_name is None:
        logger_name = f'{self.model.model_type}_Logger'
        
    # 创建logger对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers以避免重复日志
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 只创建文件处理器，不创建控制台处理器
    if hasattr(self, 'experiment_dir'):
        file_handler = logging.FileHandler(f'{self.experiment_dir}/training_log.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                    '%m/%d/%Y %I:%M:%S %p')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # 如果没有实验目录，则创建一个NullHandler以防止警告
        logger.addHandler(logging.NullHandler())
    
    return logger


def create_experiment_dir(self):
    """创建实验目录"""
    # 创建runs文件夹(如果不存在)
    if not os.path.exists(self.config.RUN_DIR):
        os.makedirs(self.config.RUN_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建实验目录
    experiment_dir = os.path.join(
        self.config.RUN_DIR, 
        f"{self.model.model_type}_{self.config.DATA.RAT}_{timestamp}"
    )
    os.makedirs(experiment_dir)
    os.makedirs(os.path.join(experiment_dir, "plots"))
    
    # 保存配置文件
    config_dict = {}
    
    # 递归提取嵌套配置
    def extract_config(cfg_node, target_dict):
        for k, v in cfg_node.items():
            if isinstance(v, dict) or hasattr(v, 'items'):
                target_dict[k] = {}
                extract_config(v, target_dict[k])
            else:
                target_dict[k] = v

    extract_config(self.config, config_dict)

    with open(os.path.join(experiment_dir, "config.yaml"), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return experiment_dir

def plot_result(self, results, loss_types=None):
    """绘制结果图表（实例方法）"""
    save_dir = self.experiment_dir
    target_file = self.experiment
    config = self.config
    
    # 如果未指定损失类型，默认使用这些
    if loss_types is None:
        loss_types = ["total loss", "MSE", "KLD"]
    
    # 神经元活动图
    plot_time = np.arange(0, 10, 0.1)
    plot_indexes = range(0, 100)
    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    for i in range(4):
        for j in range(4):
            truth = results["truth"][plot_indexes, i * 4 + j]
            predictions = results["predictions"][plot_indexes, i * 4 + j]
            axs[i, j].plot(plot_time, truth, label='Ground Truth', color='k')
            axs[i, j].plot(plot_time, predictions, label='Predictions', color='g')
            axs[i, j].set_ylabel('Smoothed spike counts')
            axs[i, j].set_xlabel('Times (s)')
            axs[i, j].set_title(f'Neuron {i * 4 + j + 1}', fontsize=12)
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=20)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(f'{save_dir}/plots/{target_file}_spike.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # 潜变量图
    fig, axs = plt.subplots(4, 4, figsize=(20, 15))
    for d in range(results["latent_mu"].shape[1]):
        row = d // 4
        col = d % 4
        if d < 16:
            axs[row, col].plot(plot_time, results["latent_mu"][plot_indexes, d])
    plt.savefig(f'{save_dir}/plots/{target_file}_latent.png', bbox_inches='tight')
    plt.close(fig)

    # 损失曲线图
    fig, axs = plt.subplots(len(loss_types), figsize=(15, 10))
    for i in range(min(len(loss_types), 4)):
        if len(loss_types) == 1:
            ax = axs
        else:
            ax = axs[i]
            
        ax.plot(np.arange(1, len(results['train_loss_all'])+1)/config.TRAIN.LOGS_PER_EPOCH,
                np.array(results['train_loss_all'])[:, i], label='train loss')
        ax.plot(np.arange(config.TRAIN.VAL_INTERVAL, config.TRAIN.NUM_UPDATES+1,
                          config.TRAIN.VAL_INTERVAL),
                np.array(results['val_loss_all'])[:, i], label='test loss')
        ax.set_ylabel(loss_types[i])
        
        if i == len(loss_types) - 1 or i == 3:
            ax.set_xlabel('epoch')
            ax.legend(loc="upper right")

    plt.savefig(f'{save_dir}/plots/{target_file}_learning_curve.png', bbox_inches='tight')
    plt.close(fig)

def attach_utils_to_runner(runner):
    """将工具函数附加到runner实例"""
    runner._setup_logger = lambda logger_name=None: setup_logger(runner, logger_name)
    runner._create_experiment_dir = lambda: create_experiment_dir(runner)
    runner._plot_result = lambda results, loss_types=None: plot_result(runner, results, loss_types)
    return runner

