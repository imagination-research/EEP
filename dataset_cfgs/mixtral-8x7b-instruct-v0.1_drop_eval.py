from opencompass.models import HuggingFaceCausalLM

datasets = [
    [
        dict(
            abbr='drop_0',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.EMEvaluator'),
                pred_postprocessor=dict(type='gsm8k')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer', 
                    dataset_split_type="random", 
                    random_idx="./data/drop/random_index.pth", 
                    start_ratio=0, end_ratio=1.0),
                prompt_template=dict(
                    template=
                    'Text: In the county, the population was spread out with 23.50% under the age of 18, 8.70% from 18 to 24, 29.70% from 25 to 44, 24.70% from 45 to 64, and 13.30% who were 65 years of age or older.\nQuestion: How many more percent are under the age of 18 compared to the 18 to 24 group?\nAnswer: According to the text, 23.5% are under the age of 18, and 8.7% are from ages 18 to 24. 23.5%-8.7%=14.8%. So the answer is 14.8.\n\nText: Playing in their second straight Thanksgiving game, the Eagles struggled especially on defense, where they were unable to stop the much-hyped Lions offense. The worst of it all was how unproven rookie Eric Rowe was tasked with covering wide receiver Calvin Johnson, leading to Johnson catching 3 touchdowns. Stafford’s five passing touchdowns, including three of them to Johnson was too much for the Eagles to overcome and for the second consecutive time this season, the Eagles gave up 45 points in a game. With the loss, the Eagles drop to 4-7 on the season and 6-1 when playing on Thanksgiving.\nQuestion: How many TD passes did Stafford throw other than to Johnson?\nAnswer: According to the text, Stafford threw 5 TD passes, 3 of which were to Johnson. 5-3=2. So the answer is 2.\n\nText: {prompt}\nQuestion: {question}\nAnswer:',
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/drop/drop_dataset_dev.json',
            reader_cfg=dict(
                input_columns=[
                    'prompt',
                    'question',
                ],
                output_column='answers',
                test_range='[0:1529]',
                test_split='validation',
                train_split='validation'),
            type='opencompass.datasets.dropDataset'),
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
        max_seq_len=2048,
        batch_size=32,
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='</s>',
        additional=dict(expert_select_type="top", fix_expert_idx=[0, 1], routing_weights_source="max"),
        batch_padding=True,
    )
]
work_dir = './outputs/reproduce/drop'