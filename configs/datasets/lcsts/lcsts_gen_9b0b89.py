from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator
from opencompass.datasets import LCSTSDataset, lcsts_postprocess

lcsts_reader_cfg = dict(input_columns=['content'], output_column='abst')

lcsts_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate, template='阅读文章：{content}\n根据上文，给出简短的单个摘要：'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

lcsts_eval_cfg = dict(
    evaluator=dict(type=JiebaRougeEvaluator),
    pred_postprocessor=dict(type=lcsts_postprocess),
)

lcsts_datasets = [
    dict(
        type=LCSTSDataset,
        abbr='lcsts',
        path='./data/LCSTS',
        reader_cfg=lcsts_reader_cfg,
        infer_cfg=lcsts_infer_cfg,
        eval_cfg=lcsts_eval_cfg)
]
