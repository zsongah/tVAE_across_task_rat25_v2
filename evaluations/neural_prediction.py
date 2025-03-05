import torch
from configs.config import get_config
from data.dataset import Dataset  
from runners import stVAE_runner

rat = '025'
latent_dim = 6
task_1mc = '1MC'
task_2mc = '2MC'
day_1mc = '2020-07-16'
day_2mc = '2020-10-05'
test_fold = 0  
load_file_prefix = f'{rat}_{task_1mc}_{day_1mc}_{test_fold}fold_' \
                f'{latent_dim}latent_' # task MC1 settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ['stVAE'] # 用于添加结果
model_files = [f'{load_file_prefix}{model_name}.pth' for model_name in model_names] 
model_results = {name: {'result_train': None, 'result_test': None} for name in model_names}
for model_file, model_name in zip(model_files, model_names):
    config_file = model_name
    config_2mc = get_config(config_file, [
            'DATA.RAT', rat,
            'DATA.TASK', task_2mc,
            'DATA.DAY', day_2mc, # can not use number_number 
            'DATA.TEST_FOLD', test_fold,
            'MODEL.LATENT_DIM', latent_dim,
        ])
    dataset_2mc = Dataset(config_2mc, rat, day_2mc, task_2mc, test_fold, device)
    checkpoint = torch.load(model_file,map_location=device)
    if model_name == 'stVAE':
        runner = stVAE_runner(config_2mc, dataset_2mc, device) 
    runner.model.load_state_dict(checkpoint['model_state_dict'])
    runner.model.eval()
    result_test = runner.evaluate()
    model_results[model_name]['result_test'] = result_test

