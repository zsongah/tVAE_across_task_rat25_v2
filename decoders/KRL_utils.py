import numpy as np
import matplotlib.pyplot as plt

def data_proprecess(latent, cue_start, press_lever, release_lever, trial_types):
    """
    处理 latent 数据以提取 rest 和 press 阶段的特征，并生成对应的标签。

    参数：
    - latent: ndarray，形状为 (108, 44, 2)，每个 trial 的特征。
    - cue_start: int，表示 rest 阶段的结束索引。
    - press_lever: int，表示 press 阶段的开始索引。
    - release_lever: int，表示 press 阶段的结束索引。
    - trial_types: list，长度为 108，表示每个 trial 的类型（1 或 -1）。

    返回：
    - data: ndarray，拼接后的 rest 和 press 数据，形状为 (总样本数, 2)。
    - labels: ndarray，对应的标签，形状为 (总样本数,)。
    """

     # 初始化存储数据和标签的列表
    processed_data = []
    processed_labels = []

    for trial_idx  in range(latent.shape[0]):
        trial = latent[trial_idx]

        rest_data = trial[:cue_start,:]
        press_data = trial[press_lever:release_lever,:]
        trial_combined_data = np.concatenate((rest_data, press_data), axis=0)
        processed_data.append(trial_combined_data)

        # 构建对应的标签
        rest_labels = [0] * rest_data.shape[0]
        press_labels = [1 if trial_types[trial_idx] == 1 else 2] * press_data.shape[0]

        # 添加到标签列表
        processed_labels.extend(rest_labels + press_labels)

    processed_data = np.vstack(processed_data)  # 合并所有 trial 的数据
    processed_labels = np.array(processed_labels)

    return processed_data, processed_labels


def plot_past_sliding_window_accuracy(true_labels, predicted_labels, window_size,save_path):
    """
    绘制过去滑窗点数的正确率曲线。
    
    Args:
        true_labels (np.ndarray): 真实标签，形状为 (n,)
        predicted_labels (np.ndarray): 预测标签，形状为 (n,)
        window_size (int): 滑窗大小
    Returns:
        None
    """
    # 确保输入的 true_labels 和 predicted_labels 是 numpy 数组
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # 检查输入长度是否一致
    assert true_labels.shape == predicted_labels.shape, "true_labels and predicted_labels must have the same shape"
    # 计算每个位置的准确性（1 表示正确，0 表示错误）
    correct = (true_labels == predicted_labels).astype(int)
    # 滑窗正确率存储
    accuracies = []

    # 遍历数组，计算过去滑窗点数的正确率
    for i in range(len(correct)):
        if i < window_size - 1:
            # 如果滑窗不足，计算从头到当前位置的正确率
            accuracies.append(np.mean(correct[:i + 1]))
        else:
            # 计算过去 window_size 个点的正确率
            accuracies.append(np.mean(correct[i - window_size + 1:i + 1]))

    # 绘制正确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label="Sliding Window Accuracy (Past 100 Points)", color="blue")
    plt.axhline(y=0.5, color="red", linestyle="--", label="Chance Level (50%)")  # 添加基线
    plt.xlabel("Index")
    plt.ylabel("Accuracy")
    plt.title(f"Sliding Window Accuracy (Window Size = {window_size})")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(save_path)
    