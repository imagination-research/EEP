compassbench_v1_language_names = [
    # ['information_retrieval_en', 'score'],
    # ['information_retrieval_zh', 'score'],
    ['intention_recognition_en_circular', 'acc_origin'],
    ['intention_recognition_en_circular', 'perf_circular'],
    ['intention_recognition_zh_circular', 'acc_origin'],
    ['intention_recognition_zh_circular', 'perf_circular'],
    ['sentiment_analysis_en_circular', 'acc_origin'],
    ['sentiment_analysis_en_circular', 'perf_circular'],
    ['sentiment_analysis_zh_circular', 'acc_origin'],
    ['sentiment_analysis_zh_circular', 'perf_circular'],
    ['translation', 'score'],
    ['content_critic_en_circular', 'acc_origin'],
    ['content_critic_en_circular', 'perf_circular'],
    ['content_critic_zh_circular', 'acc_origin'],
    ['content_critic_zh_circular', 'perf_circular'],
    ['content_summarization_en', 'rouge1'],
    ['content_summarization_zh', 'rouge1'],
    ['traditional_cultural_understanding_zh_circular', 'acc_origin'],
    ['traditional_cultural_understanding_zh_circular', 'perf_circular'],
    ['chinese_semantic_understanding_zh_circular', 'acc_origin'],
    ['chinese_semantic_understanding_zh_circular', 'perf_circular'],
]

compassbench_v1_language_groups = [
    {'name': 'language_zh_acc_1_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_zh' in name and metric != 'perf_circular']},
    {'name': 'language_en_acc_1_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_en' in name and metric != 'perf_circular']},
    {'name': 'language_acc_1_and_non_mcq', 'subsets': ['language_zh_acc_1_and_non_mcq', 'language_en_acc_1_and_non_mcq']},

    {'name': 'language_zh_perf_4_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_zh' in name and metric != 'acc_origin']},
    {'name': 'language_en_perf_4_and_non_mcq', 'subsets': [[name, metric] for name, metric in compassbench_v1_language_names if '_en' in name and metric != 'acc_origin']},
    {'name': 'language_perf_4_and_non_mcq', 'subsets': ['language_zh_perf_4_and_non_mcq', 'language_en_perf_4_and_non_mcq']},
]

summarizer = dict(
    dataset_abbrs=[
        'language_perf_4_and_non_mcq',
        'language_zh_perf_4_and_non_mcq',
        'language_en_perf_4_and_non_mcq',
        ['intention_recognition_zh_circular', 'perf_circular'],
        ['intention_recognition_en_circular', 'perf_circular'],
        ['sentiment_analysis_zh_circular', 'perf_circular'],
        ['sentiment_analysis_en_circular', 'perf_circular'],
        ['translation', 'score'],
        ['content_critic_zh_circular', 'perf_circular'],
        ['content_critic_en_circular', 'perf_circular'],
        ['content_summarization_zh', 'rouge1'],
        ['content_summarization_en', 'rouge1'],
        ['traditional_cultural_understanding_zh_circular', 'perf_circular'],
        ['chinese_semantic_understanding_zh_circular', 'perf_circular'],
    ],
    summary_groups=compassbench_v1_language_groups,
)
