from configs.config import get_config
from data.dataset import Dataset
import torch
import numpy as np
import random
from runners.stVAE_runner import stVAE_runner

def run_exp(config, dataset, proj_name_1mc):
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    experiment = f'{config.DATA.RAT}_{config.DATA.TASK}_{config.DATA.DAY}_{config.DATA.TEST_FOLD}fold_' \
                 f'{config.MODEL.LATENT_DIM}latent_' \
                 f'{proj_name_1mc}'

    print(f'Experiment {experiment} start.')
    if proj_name_1mc == 'stVAE':
        runner = stVAE_runner(config=config, 
                        dataset=dataset,
                        experiment_name = experiment,
                        )
    runner.run()
    print('\n')

def main():
    
    # '''
    # 1MC 2020-08-31; 2020-07-16
    # 2MC 2020-09-23; 2020-10-05
    # '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rat = '025'
    latent_dims = [6] 
    task_1mc = '1MC'
    day_1mc = '2020-07-16' # 数字之间加下划线不识别

    proj_name_1mc = 'stVAE'

    config_file = proj_name_1mc
    for test_fold in [0]:
        for latent_dim in latent_dims:  # [6]:
            config_1mc = get_config(config_file, [
                'DATA.RAT', rat,
                'DATA.TASK', task_1mc,
                'DATA.DAY', day_1mc, # can not use number_number 
                'DATA.TEST_FOLD', test_fold,
                'MODEL.LATENT_DIM', latent_dim,
            ])
            dataset_1mc = Dataset(config_1mc, rat, day_1mc, task_1mc, test_fold, device)
            run_exp(config_1mc, dataset_1mc, proj_name_1mc)

if __name__ == '__main__':
    main()






