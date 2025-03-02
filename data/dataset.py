import data.data_preparation as dataPrepare
import matplotlib.pyplot as plt
class Dataset:

    def __init__(self, config, rat, date, task, test_fold, device, log=True):
        super(Dataset, self).__init__()

        self.rat = rat
        self.test_fold = test_fold
        self.task = task
        self.device = device # device 在 dataset中定义，rather than in config.py

        data_preprocessed = dataPrepare.preprocess_data(f'/home/zsongah/tVAE_across_task_rat25_v2/data/raw/Rat_{self.rat}_{date[0:4]}_{date[5:7]}_{date[8:10]}_{task}_online_spk.mat',
                                           config,
                                           task,
                                           smoothed_spike=True)
        data_before_segment = dataPrepare.prepare_train_test(data_preprocessed, self.test_fold, task, log)
        data = dataPrepare.segment_all(data_before_segment, config.MODEL.TIME_WINDOW,
                                       config.TRAIN.STEP_SIZE, config.TRAIN.STEP_SIZE_TEST)

        self.data_preprocessed = data_preprocessed
        self.data_before_segment = data_before_segment # dict cannot to device

        self.train_in = data['M1_train'].to(self.device) # to(device) should be tensor
        self.test_in = data['M1_test'].to(self.device)
        self.train_out = data['M1_train'].to(self.device)
        self.test_out = data['M1_test'].to(self.device)

        self.train_movements = data['movements_train'].to(self.device)
        self.test_movements = data['movements_test'].to(self.device)
        self.train_trial_no = data['trial_No_train'].to(self.device)
        self.test_trial_no = data['trial_No_test'].to(self.device)
        self.train_events = data['events_train'].to(self.device)
        self.test_events = data['events_test'].to(self.device)
        self.train_actions = data['actions_train'].to(self.device)
        self.test_actions = data['actions_test'].to(self.device)

        self.in_neuron_num = self.train_in.size(2)
        self.out_neuron_num = self.train_out.size(2)
