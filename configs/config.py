from typing import List, Optional, Union
from yacs.config import CfgNode
import os

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

_C = CfgNode(new_allowed=True) # 创建一个名为_C的配置节点(config node)的语句
_C.SEED = 1 # previous 42
_C.RESULT_DIR = "results/"
_C.FIG_DIR = "figs/"
# _C.EVA_DIR = "evaluation/"

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