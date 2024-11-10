from opencompass.models import HuggingFaceCausalLM

datasets = [
    [
        dict(
            abbr='gsm8k_0',
            eval_cfg=dict(
                dataset_postprocessor=dict(
                    type='opencompass.datasets.gsm8k_dataset_postprocess'),
                evaluator=dict(type='opencompass.datasets.Gsm8kEvaluator'),
                pred_postprocessor=dict(
                    type='opencompass.datasets.gsm8k_postprocess')),
            infer_cfg=dict(
                inferencer=dict(
                    # max_out_len=512,
                    type='opencompass.openicl.icl_inferencer.GenInferencer',
                    dataset_split_type="range", 
                    start_ratio=0, end_ratio=1.0),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            'Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n',
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n",
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n",
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                        dict(
                            prompt=
                            'For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n',
                            role='BOT'),
                        dict(
                            prompt=
                            "Question: {question}\nLet's think step by step\nAnswer:",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/gsm8k',
            reader_cfg=dict(
                input_columns=[
                    'question',
                ],
                output_column='answer',
                test_range='[0:500]'),
            type='opencompass.datasets.GSM8KDataset'),
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
        max_out_len=150,
        max_seq_len=2048,
        batch_size=32,
        run_cfg=dict(num_gpus=4, num_procs=1),
        end_str='</s>',
        additional=dict(expert_select_type="top", fix_expert_idx=[0, 1], routing_weights_source="max"),
        batch_padding=True,
    )
]
work_dir = './outputs/reproduce/gsm8k'