CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
 
DEFAULT_CFG = {
    'model_name':  'vit_small_patch16_224',
    'num_classes': 10,
    'image_size':  224,
    'batch_size':  64,
    'epochs':      15,
    'lr':          2e-5,
    'weight_decay': 0.01,
    'warmup_epochs': 2,
    'num_workers': 2,
    'seed':        42,
    'data_dir':    './data',
    'output_dir':  './outputs',
    'use_amp':     True,
}