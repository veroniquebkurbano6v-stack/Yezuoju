这段代码是针对深度学习模型（特别是基于Transformer的模型）的一些配置设置，主要涉及两个重要的优化选项：缓存和梯度检查点。

1. 缓存设置 (use_cache)：

```python
if hasattr(model, 'config'):
    try:
        model.config.use_cache = False
    except Exception:
        pass
```

- 实现原理：在Transformer等模型中，缓存（cache）用于存储之前计算的键值对（key-value pairs），以加速推理过程。
- 用途：在训练阶段，我们通常不需要缓存，因为每次前向传播都是独立的。关闭缓存可以节省内存。
- 注意事项：
    - 只在模型有config属性时才执行
    - 使用try-except处理可能的异常，确保代码的健壮性
    - 这个设置对训练很重要，但在推理时通常需要开启

2. 梯度检查点设置 (gradient checkpointing)：

```python
if training_config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
```

- 实现原理：梯度检查点是一种用时间换空间的优化技术。它不保存中间层的所有激活值，而是在反向传播时重新计算这些值。
- 用途：
    - 显著减少训练时的显存使用
    - 允许使用更大的批次大小或更深的模型
    - 特别适用于显存受限的场景
- 注意事项：
    - 只有在配置中明确启用且模型支持时才执行
    - 会增加计算时间，因为需要重新计算中间激活值
    - 同样使用try-except确保代码的健壮性
    - 不是所有模型都支持梯度检查点

这些设置通常在模型训练开始前进行配置，是深度学习训练中的重要优化手段。代码的编写方式考虑了兼容性和错误处理，确保即使某些模型不支持这些特性也不会导致程序崩溃。