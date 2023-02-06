_base_ = [
    '../_base_/models/resnet50_cifar.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=10)


# 加载预训练模型
load_from = '/data/run01/scz0abi/mmclassification/checkpoints/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'

