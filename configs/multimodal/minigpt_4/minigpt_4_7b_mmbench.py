from opencompass.multimodal.models.minigpt_4 import (
    MiniGPT4MMBenchPromptConstructor, MiniGPT4MMBenchPostProcessor)

# dataloader settings
val_pipeline = [
    dict(type='mmpretrain.torchvision/Resize',
         size=(224, 224),
         interpolation=3),
    dict(type='mmpretrain.torchvision/ToTensor'),
    dict(type='mmpretrain.torchvision/Normalize',
         mean=(0.48145466, 0.4578275, 0.40821073),
         std=(0.26862954, 0.26130258, 0.27577711)),
    dict(type='mmpretrain.PackInputs',
         algorithm_keys=[
             'question', 'category', 'l2-category', 'context', 'index',
             'options_dict', 'options', 'split'
         ])
]

dataset = dict(type='opencompass.MMBenchDataset',
               data_file='data/mmbench/mmbench_test_20230712.tsv',
               pipeline=val_pipeline)

minigpt_4_mmbench_dataloader = dict(batch_size=1,
                                    num_workers=4,
                                    dataset=dataset,
                                    collate_fn=dict(type='pseudo_collate'),
                                    sampler=dict(type='DefaultSampler',
                                                 shuffle=False))

# model settings
minigpt_4_mmbench_model = dict(
    type='minigpt-4',
    low_resource=False,
    llama_model='/path/to/vicuna-7b/',
    prompt_constructor=dict(type=MiniGPT4MMBenchPromptConstructor,
                            image_prompt='###Human: <Img><ImageHere></Img>',
                            reply_prompt='###Assistant:'),
    post_processor=dict(type=MiniGPT4MMBenchPostProcessor))

# evaluation settings
minigpt_4_mmbench_evaluator = [
    dict(type='opencompass.DumpResults',
         save_path='work_dirs/minigpt-4-7b-mmbench.xlsx')
]

minigpt_4_mmbench_load_from = '/path/to/prerained_minigpt4_7b.pth'  # noqa
