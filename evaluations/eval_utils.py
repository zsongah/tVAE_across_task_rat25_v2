import numpy as np

def align_latent_mu(actions, latent_mu):
    # 确定reaching和press-lever的平均长度
    reaching_lengths = []
    press_lever_lengths = []
    for i in range(1,len(actions)):
        if np.isnan(actions[i]) and actions[i-1] == 0:
            reaching_start = i
        elif actions[i] == 1 and np.isnan(actions[i-1]):
            reaching_end = i
            reaching_lengths.append(reaching_end - reaching_start)
        elif actions[i] == 1 and (i == len(actions) - 1 or actions[i+1] != 1):
            press_lever_lengths.append(i - reaching_end + 1)
    
    # avg_reaching_length = int(np.mean(reaching_lengths))
    avg_reaching_length = 20
    avg_press_lever_length = int(np.mean(press_lever_lengths))
    
    # 对齐latent_mu
    aligned_latent_mu = []
    trial_types = []
    trial_start = np.where(actions == 0)[0][0]
    while trial_start < len(actions):
        rest_end = np.where(actions[trial_start:] != 0)[0][0] + trial_start
        reaching_end = np.where(~np.isnan(actions[rest_end:]))[0][0] + rest_end
        press_lever_end = np.where(np.isnan(actions[reaching_end:]))[0][0] + reaching_end
        release_end = press_lever_end + 10
        
        rest_mu = latent_mu[trial_start:rest_end,:]
        reaching_mu = latent_mu[rest_end:reaching_end,:]
        press_lever_mu = latent_mu[reaching_end:press_lever_end,:]
        release_mu = latent_mu[press_lever_end:release_end,:]
        
        # 对齐reaching
        if len(reaching_mu) < avg_reaching_length:
            # 插值到目标长度
            reaching_mu = np.array([
                np.interp(
                    np.linspace(0, 1, avg_reaching_length),  # 目标点
                    np.linspace(0, 1, len(reaching_mu)),    # 原始点
                    reaching_mu[:, dim]                    # 每一列插值
                ) for dim in range(reaching_mu.shape[1])
            ]).T  # 转置回原始形状
        else:
            # indices = np.linspace(0, len(reaching_mu) - 1, avg_press_lever_length, dtype=int)
            reaching_mu = reaching_mu[-avg_reaching_length:,:] # 取后几个
        
        # 对齐press-lever
        if len(press_lever_mu) < avg_press_lever_length:
            press_lever_mu = np.array([
                np.interp(
                np.linspace(0, 1, avg_press_lever_length), 
                np.linspace(0, 1, len(press_lever_mu)), 
                press_lever_mu[:,dim]) 
            for dim in range(press_lever_mu.shape[1]) 
            ]).T  # 转置回原始形状
        else:
            indices = np.linspace(0, len(press_lever_mu) - 1, avg_press_lever_length, dtype=int)
            press_lever_mu = press_lever_mu[indices,:]
        
        # 对齐release
        release_mu = release_mu[:10,:] 
        
        aligned_trial_mu = np.concatenate((rest_mu, reaching_mu, press_lever_mu, release_mu))
        aligned_latent_mu.append(aligned_trial_mu)
        trial_types.append(actions[reaching_end])  # 记录trial类型
        
        if np.all(np.isnan(actions[release_end:])):
            break
        trial_start = np.where(~np.isnan(actions[release_end:]))[0][0] + release_end

    aligned_latent_mu = np.array(aligned_latent_mu)
    return aligned_latent_mu, len(rest_mu), avg_reaching_length, avg_press_lever_length, len(release_mu), trial_types

