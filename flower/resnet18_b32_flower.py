# 模型配置
_base_ = [
    '../_base_/models/resnet18.py', 
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/default_runtime.py']

model = dict(
    head=dict(  
        num_classes=5,
        topk=(1, )  
    ))

# 数据配置
data = dict(
    # 根据实验环境调整每个 batch_size 和 workers 数量
    samples_per_gpu=32,
    workers_per_gpu=2, 
    # 指定训练集路径
    train=dict(
        data_prefix='data/flower/train',
        ann_file='data/flower/train.txt',
        classes='data/flower/classes.txt'),

    # 指定验证集路径
    val=dict(
        data_prefix='data/flower/val',
        ann_file='data/flower/val.txt',
        classes='data/flower/classes.txt'),
)

evaluation = dict(interval=5) # 验证期间间隔

# 优化器
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# 学习率策略
lr_config = dict(
    policy='step',
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=40)

# 加载预训练模型
load_from = '/data/run01/scz0abi/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'

# Checkpoint hook 的配置文件
checkpoint_config = dict(interval=5)  # 保存间隔
