from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen-1.8b-hf',
        path="Qwen/Qwen-1_8B",
        tokenizer_path='Qwen/Qwen-1_8B',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        pad_token_id=151643,
        min_out_len=1,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
