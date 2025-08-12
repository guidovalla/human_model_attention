_base_ = ['tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py']

# model settings
model = dict(backbone=dict(num_segments=16), cls_head=dict(num_segments=16, num_classes=3))


# dataset settings
dataset_type = 'VideoDataset'
data_root = '../../DATA_ALL/_all_videos'
data_root_val = data_root
ann_file_train = '../../DATA_ALL/train4.txt'
ann_file_val = '../../DATA_ALL/val4.txt'

data_root_test = '../../DATA_ALL/_all_videos'
ann_file_test = '../../DATA_ALL/test.txt'

file_client_args = dict(io_backend='disk')
"""
model = dict(  # Definizione del modello complessivo
    backbone=dict(  # Configurazione del backbone (la parte del modello che estrae caratteristiche dalle immagini/video)
        depth=50,  # Imposta la profondità della rete ResNet a 50 strati (ResNet-50)
        norm_eval=False,  # Disabilita la normalizzazione in fase di valutazione (utile in alcuni casi di transfer learning)
        pretrained='torchvision://resnet50',  # Usa i pesi pre-addestrati della ResNet-50 disponibili su TorchVision
        shift_div=16,  # Specifica il divisore per il meccanismo di shift temporale nel TSM (modifica una frazione delle feature temporali)
        type='ResNetTSM'  # Tipo di backbone: ResNet con il modulo TSM integrato per il riconoscimento di azioni nei video
    ),
    cls_head=dict(  # Configurazione della testa di classificazione (la parte che fa la predizione finale)
        average_clips='prob',  # Media le probabilità delle clip per ottenere la predizione finale
        consensus=dict(dim=1, type='AvgConsensus'),  # Usa una media ("AvgConsensus") sui risultati delle clip per prendere la decisione finale
        dropout_ratio=0.5,  # Imposta il dropout (per prevenire overfitting) con una probabilità di 0.5
        in_channels=2048,  # Numero di canali in ingresso (derivato da ResNet-50, che ha 2048 canali in uscita)
        init_std=0.001,  # Inizializzazione standard dei pesi della testa di classificazione (0.001)
        is_shift=True,  # Indica che la testa è configurata per usare lo shift temporale (parte del TSM)
        num_classes=3,  # Numero di classi per la classificazione (in questo caso, 3 classi)
        spatial_type='avg',  # Usa una media spaziale per aggregare le caratteristiche spaziali
        type='TSMHead'  # Tipo di testa di classificazione specifica per il modello TSM
    ),
    data_preprocessor=dict(  # Configurazione per la normalizzazione dei dati in ingresso (video o immagini)
        mean=[123.675, 116.28, 103.53],  # Media per la normalizzazione dei pixel (valori comuni per le immagini RGB)
        std=[58.395, 57.12, 57.375],  # Deviazione standard per la normalizzazione dei pixel
        type='ActionDataPreprocessor'  # Tipo di preprocessore per i dati di azioni (specializzato per input video)
    ),
    test_cfg=None,  # Configurazione per la fase di test (nessuna configurazione specifica qui)
    train_cfg=None,  # Configurazione per la fase di addestramento (nessuna configurazione specifica qui)
    type='Recognizer2D'  # Tipo di modello per il riconoscimento di azioni 2D (usa frame 2D per riconoscere le azioni nei video)
)
"""
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True,
        start_index=0),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_test),
        pipeline=test_pipeline,
        test_mode=True))



val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[25, 45],
        gamma=0.1)
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)

load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb_20220831-042b1748.pth"

