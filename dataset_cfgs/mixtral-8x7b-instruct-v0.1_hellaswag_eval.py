from opencompass.models import HuggingFaceCausalLM

datasets = [
    [
        dict(
            abbr='hellaswag_0',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator'),
                pred_postprocessor=dict(
                    options='ABCD',
                    type=
                    'opencompass.utils.text_postprocessors.first_option_postprocess'
                ),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer', 
                    dataset_split_type="range", 
                    start_ratio=0, end_ratio=1.0),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "{ctx}\nQuestion: Which ending makes the most sense?\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nYou may choose from 'A', 'B', 'C', 'D'.\nAnswer:",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/hellaswag/hellaswag.jsonl',
            reader_cfg=dict(
                input_columns=[
                    'ctx',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='label',
                test_range='[0:1674]'),
            type='opencompass.datasets.hellaswagDataset_V2'),
    ],
]
eval = dict(runner=dict(task=dict()))
models = [
    dict(
        abbr='mixtral-8x7b-instruct-v0.1',
        # type=HuggingFaceCausalLM,
        type='opencompass.models.HuggingFaceCausalLM',
        path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        tokenizer_path='mistralai/Mixtral-8x7B-Instruct-v0.1',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        meta_template=dict(
            begin="<s>",
            round=[
                dict(role="HUMAN", begin='[INST]', end='[/INST]'),
                dict(role="BOT", begin="", end='</s>', generate=True),
            ],
            eos_token_id=2
        ),
        max_out_len=30,
        max_seq_len=8192,
        batch_size=32,
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='</s>',
        additional=dict(expert_select_type="top", fix_expert_idx=[0, 1], routing_weights_source="max"),
        batch_padding=True,
    )
]
work_dir = './outputs/reproduce/hellaswag'