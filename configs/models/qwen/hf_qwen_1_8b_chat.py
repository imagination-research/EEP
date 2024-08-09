from opencompass.models import HuggingFaceCausalLM

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role="BOT", begin="\n<|im_start|>assistant\n", end='<|im_end|>', generate=True),
    ],
    eos_token_id=151645,
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-1.8b-chat-hf',
        path="Qwen/Qwen-1_8B-Chat",
        tokenizer_path='Qwen/Qwen-1_8B-Chat',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        pad_token_id=151643,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
    )
]
