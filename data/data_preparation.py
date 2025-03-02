import scipy.io as scio
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
import warnings
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.ndimage import convolve1d
from matplotlib import pyplot as plt
import copy

def preprocess_data(file_path, config, task, smoothed_spike):
    
    def filter_events(events, threshold):
        filtered_events = [events[0]]
        for event in events[1:]:
            if abs(event - filtered_events[-1]) > threshold:
                filtered_events.append(event)
        return filtered_events    
    
    def sigmoid(x, factor):
        return 1 / (1 + np.exp(-factor * x))
    
    def sigmoid_curve(m, ascending=True):
        # 生成从 -6 到 6 的 m 个点
        x = np.linspace(-6, 6, m)
        # 计算 Sigmoid 函数
        y = 1 / (1 + np.exp(-x))
        
        if not ascending:
            y = 1 - y  # 反转曲线
    
        return y
    
    def movement_generator(config):
        before_cue = int(config.PREPROCESS.BEFORE_CUE/config.PREPROCESS.BIN_SIZE)
        after_cue = int(config.PREPROCESS.AFTER_CUE/config.PREPROCESS.BIN_SIZE)
        before_press = int(config.PREPROCESS.BEFORE_PRESS/config.PREPROCESS.BIN_SIZE)
        after_press = int(config.PREPROCESS.AFTER_PRESS/config.PREPROCESS.BIN_SIZE)
        after_release = int(config.PREPROCESS.AFTER_RELEASE/config.PREPROCESS.BIN_SIZE)
        movements =[]
        movements.extend([0] * (before_cue + 1))
        movements.extend(list(sigmoid(np.linspace(-after_cue, before_press, before_press + after_cue+1), 1)))
        movements.extend([1] * after_press)
        movements.extend(list(1 - sigmoid(np.linspace(-after_release/2,after_release/2,after_release+1), 1)))
        movements = np.array(movements)
        return movements
    
    def save_plot(original, smoothed, feature_index, fig_dir):
        plt.figure(figsize=(12, 6))
        # 原始数据
        plt.subplot(2, 1, 1)
        plt.plot(original, color='blue', alpha=0.7)
        plt.title('Original Data')
        plt.xlabel('Time')
        plt.ylabel('Value')
        # 平滑后的数据
        plt.subplot(2, 1, 2)
        plt.plot(smoothed, color='red', alpha=0.7)
        plt.title('Smoothed Data')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.tight_layout()
        plt.savefig(f'{fig_dir}/feature_{feature_index}_comparison.png', bbox_inches='tight')
        plt.close()

    def causal_gaussian_smoothing(bin_spk, sigma=1.0, truncate=3.0):
        """
        Apply causal Gaussian smoothing to each feature in a multi-feature time-series dataset.

        Parameters:
        - bin_spk (np.ndarray): Input data of shape (num_features, num_timepoints).
        - sigma (float): Standard deviation of the Gaussian kernel.
        - truncate (float): Truncate the filter at this many standard deviations.

        Returns:
        - smoothed_bin_spk (np.ndarray): Smoothed data with the same shape as bin_spk.
        """
        num_features, num_timepoints = bin_spk.shape
        
        # Calculate the radius of the kernel based on sigma and truncate
        radius = int(truncate * sigma + 0.5)
        
        # Create a one-sided Gaussian kernel (current and past)
        t = np.arange(0, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        
        # Normalize the kernel so that the sum equals 1
        kernel /= kernel.sum()
        
        # Initialize the smoothed data array
        smoothed_bin_spk = np.zeros_like(bin_spk)
        
        # Apply the causal Gaussian filter to each feature
        for i in range(num_features):
            # Pad the input signal on the left (past) with the edge value to maintain causality
            pad_width = radius
            x_padded = np.pad(bin_spk[i], (pad_width, 0), mode='edge')
            
            # Perform the convolution using the one-sided Gaussian kernel
            y = convolve1d(x_padded, kernel, mode='constant', cval=0.0)
            
            # Remove the padding to restore the original signal length
            smoothed_bin_spk[i] = y[pad_width:]

        return smoothed_bin_spk

    data = scio.loadmat(file_path)
    high = 1.0
    low = -1.0
    # extract event 
    success_all = filter_events(data['EVT03'][:, 0].flatten(), 1)
    cue_start_all = filter_events(data['EVT05'][:, 0].flatten(), 1)
    press_lever_all = filter_events(data['EVT06'][:, 0].flatten(), 0.01)
    release_lever_all = filter_events(data['EVT07'][:, 0].flatten(), 0.01)

    trial_time_index_whole = []
    trial_no_whole = []
    movements_whole = []
    movements_whole_y2 = []
    action = []
    events_whole = []
    trial_type = []
    for success in success_all[1:]:
        cue_start = None
        for x in reversed(cue_start_all):
            if x < success:
                cue_start = x
                break

        press_lever = None
        for x in reversed(press_lever_all):
            if x < success:
                press_lever = x
                break

        release_lever = None
        for x in release_lever_all:
            if x > success:
                release_lever = x
                break

        # print(f"Found cue_start: {cue_start}, press_lever: {press_lever}, release_lever: {release_lever}")
        if (cue_start is None or 
            press_lever is None or 
            success - press_lever > 1.0 or 
            press_lever - cue_start < config.PREPROCESS.AFTER_CUE + config.PREPROCESS.BEFORE_PRESS + 0.1):
            # print("Skipping due to invalid conditions")
            continue

        if min(abs(cue_start - data['EVT01'][:, 0].flatten())) < 1e-2:
            height = high
        else:
            height = low
        trial_type.append(height)
        # *10 for unit to 0.1s, eliminate the float error
        cue_start = int(round(cue_start * 10)) # round to 1 decimal place
        press_lever = int(round(press_lever * 10))
        success = int(round(success * 10))
        release_lever = int(round(release_lever * 10))

        trial_start = cue_start - int(config.PREPROCESS.BEFORE_CUE_LONG * 10)
        trial_end = release_lever + int(config.PREPROCESS.AFTER_RELEASE_LONG *10)
        num_elements = int((trial_end - trial_start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)

        trial_time_index_whole.append(np.linspace(trial_start, trial_end, num_elements).astype(int))
        trial_no_whole.append(np.array([len(trial_time_index_whole)] * num_elements).astype(int))
        movements_whole.append(np.array([float('nan')] * num_elements))
        movements_whole_y2.append(np.array([float('nan')] * num_elements))
        # action：cue 前5个点 0， press-release high 1， press-release low -1
        action.append(np.array([float('nan')] * num_elements))
        # 0 for rest, 1 for high press, 2 for high release, -1 for low press, -2 for low release
        events_whole.append(np.array([float('nan')] * num_elements)) 
        # rest
        trial_time_index = []
        start = cue_start - int(config.PREPROCESS.BEFORE_CUE * 10) 
        stop = cue_start - 1
        num_elements = int((stop - start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)
        rest_idx = np.linspace(start, stop, num_elements)
        trial_time_index.extend(rest_idx)
        movements_whole[-1][np.where(np.isin(trial_time_index_whole[-1],rest_idx))[0]] = 0
        movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],rest_idx))[0]] = 0
        events_whole[-1][np.where(np.isin(trial_time_index_whole[-1],cue_start))[0]] = 0

        # reaching
        start = cue_start + 1 
        stop = cue_start + int(config.PREPROCESS.AFTER_CUE * 10)
        num_elements = int((stop - start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)
        reach_pre_idx = np.linspace(start, stop, num_elements)
        start = press_lever - int(config.PREPROCESS.BEFORE_PRESS * 10)
        stop = press_lever - 1
        num_elements = int((stop - start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)
        reach_pos_idx = np.linspace(start, stop, num_elements)        
        reach_idx = np.concatenate((reach_pre_idx, reach_pos_idx))
        movements_whole[-1][np.where(np.isin(trial_time_index_whole[-1],reach_idx))[0]] = sigmoid_curve(len(reach_idx))
        if height == high:
            movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],reach_idx))[0]] = sigmoid_curve(len(reach_idx))
        else:
            movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],reach_idx))[0]] = -1*sigmoid_curve(len(reach_idx))

        # press and hold
        start = press_lever
        stop = press_lever + int(config.PREPROCESS.AFTER_PRESS*10)
        num_elements = int((stop - start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)
        press_idx = np.linspace(start, stop, num_elements)
        movements_whole[-1][np.where(np.isin(trial_time_index_whole[-1],press_idx))[0]] = 1
        if height == high:
            movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],press_idx))[0]] = 1
            events_whole[-1][np.where(np.isin(trial_time_index_whole[-1],press_lever))[0]] = 1
        else:
            movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],press_idx))[0]] = -1
            events_whole[-1][np.where(np.isin(trial_time_index_whole[-1],press_lever))[0]] = -1
        
        # release
        start = release_lever + 1
        stop = release_lever + int(config.PREPROCESS.AFTER_RELEASE * 10)
        num_elements = int((stop - start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)
        release_idx = np.linspace(start, stop, num_elements)
        movements_whole[-1][np.where(np.isin(trial_time_index_whole[-1],release_idx))[0]] = sigmoid_curve(len(release_idx), ascending=False)
        if height == high:
            movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],release_idx))[0]] = sigmoid_curve(len(release_idx), ascending=False)
            events_whole[-1][np.where(np.isin(trial_time_index_whole[-1],release_lever))[0]] = 2
        else:
            movements_whole_y2[-1][np.where(np.isin(trial_time_index_whole[-1],release_idx))[0]] = -1*sigmoid_curve(len(release_idx), ascending=False)
            events_whole[-1][np.where(np.isin(trial_time_index_whole[-1],release_lever))[0]] = -2
    
        # action：cue 前5个点 0， press-release high 1， press-release low -1
        action[-1][np.where(np.isin(trial_time_index_whole[-1],np.linspace(cue_start-5,cue_start,6)))[0]] = 0
        start = press_lever
        stop = release_lever
        num_elements = int((stop - start) / int(config.PREPROCESS.BIN_SIZE*10) + 1)
        if height == high:
            action[-1][np.where(np.isin(trial_time_index_whole[-1],np.linspace(start,stop,num_elements)))[0]] = 1 
        else:
            action[-1][np.where(np.isin(trial_time_index_whole[-1],np.linspace(start,stop,num_elements)))[0]] = -1
    
    # spikes
    n_bin = trial_time_index_whole[-1][-1]
    bin_spk = np.zeros((32, n_bin))
    combined_spike = {i: [] for i in range(32)}
    for i_neuron in range(0, 32):
        for suffix in ['a', 'b', 'U']:
            key = f'SPK{i_neuron+1:02d}{suffix}'
            if key in data:
                combined_spike[i_neuron].extend(data[key].flatten())
        combined_spike[i_neuron] = sorted(combined_spike[i_neuron])
        end_point = round(n_bin / 10 + config.PREPROCESS.BIN_SIZE, 1)
        bin_spk[i_neuron,:] =  np.histogram(combined_spike[i_neuron], bins=np.arange(0, end_point, config.PREPROCESS.BIN_SIZE))[0]
    
        # causal caussion makes a delay
    if smoothed_spike:
        bin_spk = gaussian_filter1d(bin_spk, sigma=1.5) # causal_gaussian_filter1d
        # save_plot(bin_spk[0,:200], bin_spk_smoothed[0,:200], 0, './figs')

    bin_spk_concat = np.empty((bin_spk.shape[0], 0))
    movements = np.array([])
    movements_y2 = np.array([]) # for 2MC task
    trial_no = np.array([],dtype=int)
    events = np.array([],dtype=int)
    actions = np.array([])
    for i in range(len(trial_time_index_whole)):
        bin_spk_concat = np.hstack((bin_spk_concat, bin_spk[:, trial_time_index_whole[i].astype(int)-1]))
        movements = np.hstack((movements, movements_whole[i]))
        trial_no = np.hstack((trial_no, trial_no_whole[i]))
        events = np.hstack((events, events_whole[i]))
        actions = np.hstack((actions, action[i]))
        # if task == "2MC":
        movements_y2 = np.hstack((movements_y2, movements_whole_y2[i]))
    # if task == "2MC":
    movements = np.vstack((movements, movements_y2))
    trial_type = np.array(trial_type, dtype=int)
    numbers = np.arange(1, len(trial_time_index_whole) + 1)
    indices = np.arange(len(numbers))
    # np.random.shuffle(indices) # no shuffle
    numbers_shuffled = numbers[indices]
    trial_type_shuffled = trial_type[indices]
    fold_size = len(numbers_shuffled) // 5
    folds = [numbers_shuffled[i * fold_size: (i + 1) * fold_size] for i in range(4)]
    folds.append(numbers_shuffled[4 * fold_size:])  # Include the remaining elements in the last fold
    trial_type_folds = [trial_type_shuffled[i * fold_size: (i + 1) * fold_size] for i in range(4)]
    trial_type_folds.append(trial_type_shuffled[4 * fold_size:])  # Include the remaining elements in the last fold

    # convert it into torch
    bin_spk_concat = torch.from_numpy(bin_spk_concat).float()
    trial_no = torch.from_numpy(trial_no).int()
    events = torch.from_numpy(events).int()
    actions = torch.from_numpy(actions).float()
    movements = torch.from_numpy(movements).float()

    folds = [torch.from_numpy(fold) for fold in folds]
    trial_type_folds = [torch.from_numpy(fold) for fold in trial_type_folds]

    ## free-moving data construction
    mask = np.zeros(bin_spk.shape[1], dtype=bool)
    for indices in trial_time_index_whole:
        mask[indices.astype(int) - 1] = True
    bin_spk_free_moving = bin_spk[:16, ~mask] # M1
    bin_spk_free_moving = torch.from_numpy(bin_spk_free_moving).float()
    return_dict = {
        'spikes': bin_spk_concat, 
        'spikes_free_moving': bin_spk_free_moving,
        'movements': movements,
        'trial_No': trial_no,
        'events': events,
        'actions': actions,
        'folds': folds,
        'trial_type_folds': trial_type_folds
    }

    return return_dict  

def prepare_train_test(data, test_fold, task, log=True):
    test_trials = data["folds"][test_fold]
    train_trials = torch.cat([data["folds"][fold] for fold in range(5) if fold != test_fold])
    test_type = data["trial_type_folds"][test_fold]
    train_type = torch.cat([data["trial_type_folds"][fold] for fold in range(5) if fold != test_fold])

    train_indices = torch.cat([torch.nonzero(torch.eq(data["trial_No"], i)) for i in train_trials]).squeeze()
    test_indices = torch.cat([torch.nonzero(torch.eq(data["trial_No"], i)) for i in test_trials]).squeeze()
    
    m1_mpfc_train = data["spikes"][:, train_indices]
    m1_mpfc_test = data["spikes"][:, test_indices]
    m1_train = data["spikes"][:16, train_indices]
    m1_test = data["spikes"][:16, test_indices]

    # if task == "1MC":
    #     movements_train = data["movements"][train_indices].unsqueeze(0)# adds a new dimension
    #     movements_test = data["movements"][test_indices].unsqueeze(0)# adds a new dimension
    # else:
    movements_train = data["movements"][:,train_indices]
    movements_test = data["movements"][:,test_indices]
    
    trial_no_train = data["trial_No"][train_indices].unsqueeze(0) # adds a new dimension
    trial_no_test = data["trial_No"][test_indices].unsqueeze(0)
    events_train = data["events"][train_indices].unsqueeze(0)
    events_test = data["events"][test_indices].unsqueeze(0)
    actions_train = data["actions"][train_indices].unsqueeze(0)
    actions_test = data["actions"][test_indices].unsqueeze(0)


    if log: # .size是torch的命令
        print(f'| load neural data | '
              f'train length: {m1_train.size(1)}, trial: {len(train_trials)} | '
              f'test length: {m1_test.size(1)} , trial: {len(test_trials)} | '
              f'{m1_mpfc_train.size(0)} total neurons | '
              f'{m1_train.size(0)} M1 neurons | ')

    return_dict = {
        'M1_mPFC_train': m1_mpfc_train,
        'M1_train': m1_train,
        'movements_train': movements_train,
        'trial_No_train': trial_no_train, # trial index for each time point
        'events_train': events_train,
        'actions_train': actions_train,
        'M1_mPFC_test': m1_mpfc_test,
        'M1_test': m1_test,
        'movements_test': movements_test,
        'trial_No_test': trial_no_test,# trial index for each time point
        'events_test': events_test,
        'actions_test': actions_test,
        'train_trials': train_trials, #trial number
        'test_trials': test_trials,  #trial number
        'train_type': train_type,
        'test_type': test_type
    }

    return return_dict

def segment_within_trial(input_data, seq_len, step_size, trial_idx):
    """Process the data into segments within trials.

    Arguments:
        input_data: Tensor, shape ``[neuron_num, N]``
        seq_len: int, time window size
        step_size: int, interval between the beginning of two sequences
        trial_idx: Tensor, shape ``[N]``, indicating trial numbers

    Returns:
        Tensor, shape ``[seq_len + 1, segment_num, neuron_num]``
    """
    neuron_num, total_len = input_data.shape
    all_segments = []

    # 找到每个 trial 的唯一索引
    unique_trials = torch.unique(trial_idx)

    for trial in unique_trials:
        trial_indices = torch.nonzero(trial_idx == trial, as_tuple=True)[1]
        data = input_data[:, trial_indices]
        all_segments.append(segment(data, seq_len, step_size))
        
    return torch.cat(all_segments, dim=1)

def segment(input_data, seq_len, step_size):
    """Process the data into segments with overlapping

    Arguments:
        input_data: Tensor, shape ``[neuron_num, N]``
        seq_len: int, time window size
        step_size: interval between the beginning of two sequences

    Returns:
        Tensor, shape ``[seq_len, segment_num, neuron_num]``
    """

    neuron_num, total_len = input_data.size()
    segment_num = (total_len - seq_len - 1) // step_size + 1
    segments = np.empty((seq_len + 1, segment_num, neuron_num))

    for seq, i in enumerate(range(0, total_len - seq_len - 1, step_size)): # seq_len = 300
        segments[:, seq, :] = input_data[:, i: i + seq_len + 1].t()
    
    if (total_len - seq_len - 1) % step_size == 0:
        segments[:, seq+1,:] = input_data[:, total_len - seq_len - 1: total_len].t()

    return torch.FloatTensor(segments)

def segment_all(data, time_window, train_step_size, test_step_size):
    # Create a deep copy of the data to avoid modifying the original input
    data_copy = copy.deepcopy(data)
    data_copy['M1_mPFC_train'] = segment_within_trial(data['M1_mPFC_train'], time_window, train_step_size, data['trial_No_train']) #10 
    data_copy['M1_train'] = segment_within_trial(data['M1_train'], time_window, train_step_size, data['trial_No_train'])
    data_copy['movements_train'] = segment_within_trial(data['movements_train'], time_window, train_step_size, data['trial_No_train'])
    data_copy['trial_No_train'] = segment_within_trial(data['trial_No_train'], time_window, train_step_size, data['trial_No_train'])
    data_copy['events_train'] = segment_within_trial(data['events_train'], time_window, train_step_size, data['trial_No_train'])
    data_copy['actions_train'] = segment_within_trial(data['actions_train'], time_window, train_step_size, data['trial_No_train'])

    data_copy['M1_mPFC_test'] = segment_within_trial(data['M1_mPFC_test'], time_window, test_step_size,data['trial_No_test']) #200
    data_copy['M1_test'] = segment_within_trial(data['M1_test'], time_window, test_step_size,data['trial_No_test'])
    data_copy['movements_test'] = segment_within_trial(data['movements_test'], time_window, test_step_size,data['trial_No_test'])
    data_copy['trial_No_test'] = segment_within_trial(data['trial_No_test'], time_window, test_step_size,data['trial_No_test'])
    data_copy['events_test'] = segment_within_trial(data['events_test'], time_window, test_step_size,data['trial_No_test'])
    data_copy['actions_test'] = segment_within_trial(data['actions_test'], time_window, test_step_size,data['trial_No_test'])

    return data_copy

def get_batch(segments: Tensor, bsz: int, i: int) -> Tuple[Tensor, Tensor]:
    warnings.warn("Use get_batch_random for train and get_batch_ss for test.", DeprecationWarning)
    """
    Args:
        segments: Tensor, shape ``[seq_len, segment_num, neuron_num]``
        bsz: int, batch size
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size, neuron_num]``
        and target has shape ``[seq_len * batch_size, neuron_num]``
    """
    neuron_num = segments.size(2)
    data = segments[0:-1, i:i + bsz]  
    target = segments[1:, i:i + bsz].permute(1, 0, 2).reshape(-1, neuron_num)
    return data, target


def get_batch_ss(segments: Tensor, bsz: int, i: int, step_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """
    ss is short for step size
    Args:
        segments: Tensor, shape ``[seq_len, segment_num, neuron_num]``
        bsz: int, batch size
        i: int, start position of batch
        step_size: int, determine the length of target

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size, neuron_num]``
        and target has shape ``[seq_len * batch_size, neuron_num]``
    """
    segment_num = segments.size(1)
    neuron_num = segments.size(2)
    data = segments[0:-1, i:min(i + bsz, segment_num)]
    target = segments[1:, i:min(i + bsz, segment_num)].permute(1, 0, 2).reshape(-1, neuron_num)
    target_valid = segments[-step_size:, i:min(i + bsz, segment_num)].permute(1, 0, 2).reshape(-1, neuron_num)
    return data, target, target_valid


def get_batch_random(segments: Tensor, bsz: int, indices: list, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        segments: Tensor, shape ``[seq_len, segment_num, neuron_num]``
        bsz: int, batch size
        indices: list, a list of shuffled indices
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size, neuron_num]``
        and target has shape ``[seq_len * batch_size, neuron_num]``
    """
    segment_num = segments.size(1)
    neuron_num = segments.size(2)

    indices = indices[i:min(i + bsz, segment_num)]

    data = segments[0:-1, indices]
    target = segments[1:, indices].permute(1, 0, 2).reshape(-1, neuron_num)
    return data, target


def process_trials(M1_test, movements_test, trial_No_test, test_trial):
    """Process the data into segments based on movements and trials.

    Arguments:
        M1_test: Tensor, shape [N, T], feature data
        movements_test: Tensor, shape [T], movement data (with NaNs)
        trial_No_test: Tensor, shape [T], trial indices
        test_trial: Tensor, shape [num_trials], unique trial numbers to process

    Returns:
        Tensor, shape [seq_len, batch, N], structured data
    """
    
    # 检查输入的形状
    N, T = M1_test.shape
    num_trials = test_trial.size(0)

    # 存储最终的 segments
    data_segments = []
    movements_segments = []
    trial_No_segments = []

    for trial in test_trial:
        # 找到当前 trial 的索引
        trial_indices = torch.nonzero(trial_No_test == trial, as_tuple=True)[1] # trial_No_test 2d tensor      
        non_nan_indices = torch.nonzero(~torch.isnan(movements_test), as_tuple=True)[1]
        valid_indices = trial_indices[torch.isin(trial_indices, non_nan_indices)]
        data_segments.append(M1_test[:,valid_indices])
        movements_segments.append(movements_test[:,valid_indices])
        trial_No_segments.append(trial_No_test[:,valid_indices])

    seq_len = data_segments[0].shape[1]
    batch_num = len(data_segments)
    data = torch.stack(data_segments, dim=1)
    data = data.permute(2, 1, 0)
    movements = torch.stack(movements_segments, dim=1)
    movements = movements.permute(2, 1, 0)
    trial_No = torch.stack(trial_No_segments, dim=1)
    trial_No = trial_No.permute(2, 1, 0)

    return data, movements, trial_No
