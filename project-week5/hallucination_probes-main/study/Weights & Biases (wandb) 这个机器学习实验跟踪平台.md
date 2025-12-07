## wandb简介

Weights & Biases 是一个用于机器学习实验跟踪、数据集版本控制和模型管理的平台。它可以帮助研究人员和开发者更好地组织和比较他们的实验

## 代码功能解析

wandb.init(project=training_config.wandb_project, name=training_config.probe_config.probe_id)

1. wandb.init(): 这是 wandb 的初始化函数，用于开始一个新的实验运行
2. project=training_config.wandb_project: 指定这个实验属于哪个项目。项目名来自配置对象 training_config 中的 wandb_project
   属性
3. name=training_config.probe_config.probe_id: 为这次特定的实验运行指定一个名称。这个名称来自配置中的 probe_config 的
   probe_id
4. wandb.init() 函数会返回一个 wandb.run 对象，这个对象可以用来记录实验的指标、日志和其它信息