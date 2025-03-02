project_root/
├── configs/
│   └── tVAE.yaml          # 你的方法专属配置
├── data/
│   ├── __init__.py
│   ├── dataset.py         # 统一数据接口（需改造）
│   └── preparation.py     # 原data_preparation_realData_suc功能
├── models/
│   ├── my_method/         # 你的方法核心实现
│   │   ├── __init__.py
│   │   ├── model.py       # 原VAE模型
│   │   ├── runner.py      # 原Runner类改造
│   │   └── modules/       # 原network_modules内容
│   │       ├── encoder.py
│   │       ├── decoder.py
│   │       └── positional.py
├── main.py                # 主入口（原run.py改造）
└── evaluation/

├── results/
└── metrics.py         # 评估指标
