from typing import List, Optional, Union
from yacs.config import CfgNode
import os

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

_C = CfgNode(new_allowed=True) # 创建一个名为_C的配置节点(config node)的语句
_C.SEED = 1 # previous 42
_C.RESULT_DIR = "results/"
_C.FIG_DIR = "figs/"
_C.RUN_DIR = "runs/"

_C.DATA = CfgNode(new_allowed=True)
_C.DATA.RAT = ""         # 示例: 实验动物或对象标识
_C.DATA.TEST_FOLD = 0             # 用于交叉验证等
_C.DATA.DAY = ""        # 日期标识
_C.DATA.TASK = ""     # 数据任务类型

_C.PREPROCESS = CfgNode(new_allowed=True)
_C.PREPROCESS.BIN_SIZE = 0.1
_C.PREPROCESS.BEFORE_CUE_LONG = 2       # seconds, 训练时使用的时间段
_C.PREPROCESS.BEFORE_CUE = 0            # cue前的时间点（不包含cue时刻）
_C.PREPROCESS.AFTER_CUE = 0           # cue后的时间段
_C.PREPROCESS.BEFORE_PRESS = 1          # press前的时间段
_C.PREPROCESS.AFTER_PRESS = 0.5         # press后立即的时间段
_C.PREPROCESS.AFTER_RELEASE = 1         # release后不包括该点的时间段
_C.PREPROCESS.AFTER_RELEASE_LONG = 2    # release后的长时间段（训练时需要）

def get_cfg_defaults():
    """Get default config (yacs config node)."""
    return _C.clone()

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CfgNode:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths..
    :param opts: Config options (keys, values) in a list, e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(DEFAULT_CONFIG_DIR + config_path + '.yaml')

    if opts:
        # for i in range(0, len(opts), 2):
        #     key, value = opts[i], opts[i + 1]
        #     print(f"Config option: {key} = {value} ({type(value)})")
        config.merge_from_list(opts)

    config.freeze()
    return config