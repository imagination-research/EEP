<div align="center">
  <img src="docs/zh_cn/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![docs](https://readthedocs.org/projects/opencompass/badge)](https://opencompass.readthedocs.io/zh_CN)
[![license](https://img.shields.io/github/license/InternLM/opencompass.svg)](https://github.com/open-compass/opencompass/blob/main/LICENSE)

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐官方网站](https://opencompass.org.cn/) |
[📖数据集社区](https://hub.opencompass.org.cn/home) |
[📊性能榜单](https://rank.opencompass.org.cn/home) |
[📘文档教程](https://opencompass.readthedocs.io/zh_CN/latest/index.html) |
[🛠️安装](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[🤔报告问题](https://github.com/open-compass/opencompass/issues/new/choose)

[English](/README.md) | 简体中文

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> 和 <a href="https://r.vansin.top/?r=opencompass" target="_blank">微信社区</a>
</p>

## 📣 OpenCompass 2.0

我们很高兴发布 OpenCompass 司南 2.0 大模型评测体系，它主要由三大核心模块构建而成：[CompassKit](https://github.com/open-compass)、[CompassHub](https://hub.opencompass.org.cn/home)以及[CompassRank](https://rank.opencompass.org.cn/home)。

**CompassRank** 系统进行了重大革新与提升，现已成为一个兼容并蓄的排行榜体系，不仅囊括了开源基准测试项目，还包含了私有基准测试。此番升级极大地拓宽了对行业内各类模型进行全面而深入测评的可能性。

**CompassHub** 创新性地推出了一个基准测试资源导航平台，其设计初衷旨在简化和加快研究人员及行业从业者在多样化的基准测试库中进行搜索与利用的过程。为了让更多独具特色的基准测试成果得以在业内广泛传播和应用，我们热忱欢迎各位将自定义的基准数据贡献至CompassHub平台。只需轻点鼠标，通过访问[这里](https://hub.opencompass.org.cn/dataset-submit)，即可启动提交流程。

**CompassKit** 是一系列专为大型语言模型和大型视觉-语言模型打造的强大评估工具合集，它所提供的全面评测工具集能够有效地对这些复杂模型的功能性能进行精准测量和科学评估。在此，我们诚挚邀请您在学术研究或产品研发过程中积极尝试运用我们的工具包，以助您取得更加丰硕的研究成果和产品优化效果。

## 🧭	欢迎

来到**OpenCompass**！

就像指南针在我们的旅程中为我们导航一样，我们希望OpenCompass能够帮助你穿越评估大型语言模型的重重迷雾。OpenCompass提供丰富的算法和功能支持，期待OpenCompass能够帮助社区更便捷地对NLP模型的性能进行公平全面的评估。

🚩🚩🚩 欢迎加入 OpenCompass！我们目前**招聘全职研究人员/工程师和实习生**。如果您对 LLM 和 OpenCompass 充满热情，请随时通过[电子邮件](mailto:zhangsongyang@pjlab.org.cn)与我们联系。我们非常期待与您交流！

🔥🔥🔥 祝贺 **OpenCompass 作为大模型标准测试工具被Meta AI官方推荐**, 点击 Llama 的 [入门文档](https://ai.meta.com/llama/get-started/#validation) 获取更多信息.

> **注意**<br />
> 我们正式启动 OpenCompass 共建计划，诚邀社区用户为 OpenCompass 提供更具代表性和可信度的客观评测数据集!
> 点击 [Issue](https://github.com/open-compass/opencompass/issues/248) 获取更多数据集.
> 让我们携手共进，打造功能强大易用的大模型评测平台！

## 🚀 最新进展 <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2024.02.29\]** 我们支持了MT-Bench、AlpacalEval和AlignBench，更多信息可以在[这里](https://opencompass.readthedocs.io/en/latest/advanced_guides/subjective_evaluation.html)找到 🔥🔥🔥。
- **\[2024.01.30\]** 我们发布了OpenCompass 2.0。更多信息，请访问[CompassKit](https://github.com/open-compass)、[CompassHub](https://hub.opencompass.org.cn/home)和[CompassRank](https://rank.opencompass.org.cn/home) 🔥🔥🔥。
- **\[2024.01.17\]** 我们支持了 [InternLM2](https://github.com/open-compass/opencompass/blob/main/configs/eval_internlm2_chat_keyset.py) 和 [InternLM2-Chat](https://github.com/open-compass/opencompass/blob/main/configs/eval_internlm2_chat_keyset.py) 的相关评测，InternLM2 在这些测试中表现出非常强劲的性能，欢迎试用！🔥🔥🔥.
- **\[2024.01.17\]** 我们支持了多根针版本的大海捞针测试，更多信息见[这里](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/needleinahaystack_eval.html#id8)🔥🔥🔥.
- **\[2023.12.28\]** 我们支持了对使用[LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory)（一款强大的LLM开发工具箱）开发的所有模型的无缝评估!
- **\[2023.12.22\]** 我们开源了[T-Eval](https://github.com/open-compass/T-Eval)用于评测大语言模型工具调用能力。欢迎访问T-Eval的官方[Leaderboard](https://open-compass.github.io/T-Eval/leaderboard.html)获取更多信息!

> [更多](docs/zh_cn/notes/news.md)

## ✨ 介绍

![image](https://github.com/open-compass/opencompass/assets/22607038/30bcb2e2-3969-4ac5-9f29-ad3f4abb4f3b)

OpenCompass 是面向大模型评测的一站式平台。其主要特点如下：

- **开源可复现**：提供公平、公开、可复现的大模型评测方案

- **全面的能力维度**：五大维度设计，提供 70+ 个数据集约 40 万题的的模型评测方案，全面评估模型能力

- **丰富的模型支持**：已支持 20+ HuggingFace 及 API 模型

- **分布式高效评测**：一行命令实现任务分割和分布式评测，数小时即可完成千亿模型全量评测

- **多样化评测范式**：支持零样本、小样本及思维链评测，结合标准型或对话型提示词模板，轻松激发各种模型最大性能

- **灵活化拓展**：想增加新模型或数据集？想要自定义更高级的任务分割策略，甚至接入新的集群管理系统？OpenCompass 的一切均可轻松扩展！

## 📊 性能榜单

我们将陆续提供开源模型和 API 模型的具体性能榜单，请见 [OpenCompass Leaderboard](https://rank.opencompass.org.cn/home) 。如需加入评测，请提供模型仓库地址或标准的 API 接口至邮箱  `opencompass@pjlab.org.cn`.

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🛠️ 安装

下面展示了快速安装以及准备数据集的步骤。

### 💻 环境配置

#### 面向开源模型的GPU环境

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

#### 面向API模型测试的CPU环境

```bash
conda create -n opencompass python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
# 如果需要使用各个API模型，请 `pip install -r requirements/api.txt` 安装API模型的相关依赖
```

### 📂 数据准备

```bash
# 下载数据集到 data/ 处
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

有部分第三方功能,如 Humaneval 以及 Llama,可能需要额外步骤才能正常运行，详细步骤请参考[安装指南](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🏗️ ️评测

确保按照上述步骤正确安装 OpenCompass 并准备好数据集后，可以通过以下命令评测 LLaMA-7b 模型在 MMLU 和 C-Eval 数据集上的性能：

```bash
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
```

OpenCompass 预定义了许多模型和数据集的配置，你可以通过 [工具](./docs/zh_cn/tools.md#ListConfigs) 列出所有可用的模型和数据集配置。

```bash
# 列出所有配置
python tools/list_configs.py
# 列出所有跟 llama 及 mmlu 相关的配置
python tools/list_configs.py llama mmlu
```

你也可以通过命令行去评测其它 HuggingFace 模型。同样以 LLaMA-7b 为例：

```bash
python run.py --datasets ceval_ppl mmlu_ppl \
--hf-path huggyllama/llama-7b \  # HuggingFace 模型地址
--model-kwargs device_map='auto' \  # 构造 model 的参数
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # 构造 tokenizer 的参数
--max-out-len 100 \  # 最长生成 token 数
--max-seq-len 2048 \  # 模型能接受的最大序列长度
--batch-size 8 \  # 批次大小
--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
--num-gpus 1  # 运行该模型所需的最少 gpu 数
```

> **注意**<br />
> 若需要运行上述命令，你需要删除所有从 `# ` 开始的注释。

通过命令行或配置文件，OpenCompass 还支持评测 API 或自定义模型，以及更多样化的评测策略。请阅读[快速开始](https://opencompass.readthedocs.io/zh_CN/latest/get_started/quick_start.html)了解如何运行一个评测任务。

更多教程请查看我们的[文档](https://opencompass.readthedocs.io/zh_CN/latest/index.html)。

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📖 数据集支持

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>语言</b>
      </td>
      <td>
        <b>知识</b>
      </td>
      <td>
        <b>推理</b>
      </td>
      <td>
        <b>考试</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>字词释义</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>成语习语</b></summary>

- CHID

</details>

<details open>
<summary><b>语义相似度</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>指代消解</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>翻译</b></summary>

- Flores
- IWSLT2017

</details>

<details open>
<summary><b>多语种问答</b></summary>

- TyDi-QA
- XCOPA

</details>

<details open>
<summary><b>多语种总结</b></summary>

- XLSum

</details>
      </td>
      <td>
<details open>
<summary><b>知识问答</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestions
- TriviaQA

</details>
      </td>
      <td>
<details open>
<summary><b>文本蕴含</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE
- ANLI

</details>

<details open>
<summary><b>常识推理</b></summary>

- StoryCloze
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>数学推理</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>定理应用</b></summary>

- TheoremQA
- StrategyQA
- SciBench

</details>

<details open>
<summary><b>综合推理</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>初中/高中/大学/职业考试</b></summary>

- C-Eval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC
- Xiezhi

</details>

<details open>
<summary><b>医学考试</b></summary>

- CMB

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>理解</b>
      </td>
      <td>
        <b>长文本</b>
      </td>
      <td>
        <b>安全</b>
      </td>
      <td>
        <b>代码</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>阅读理解</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE
- DROP
- OpenBookQA
- SQuAD2.0

</details>

<details open>
<summary><b>内容总结</b></summary>

- CSL
- LCSTS
- XSum
- SummScreen

</details>

<details open>
<summary><b>内容分析</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
      <td>
<details open>
<summary><b>长文本理解</b></summary>

- LEval
- LongBench
- GovReports
- NarrativeQA
- Qasper

</details>
      </td>
      <td>
<details open>
<summary><b>安全</b></summary>

- CivilComments
- CrowsPairs
- CValues
- JigsawMultilingual
- TruthfulQA

</details>
<details open>
<summary><b>健壮性</b></summary>

- AdvGLUE

</details>
      </td>
      <td>
<details open>
<summary><b>代码</b></summary>

- HumanEval
- HumanEvalX
- MBPP
- APPs
- DS1000

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 📖 模型支持

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>开源模型</b>
      </td>
      <td>
        <b>API 模型</b>
      </td>
      <!-- <td>
        <b>自定义模型</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

- [InternLM](https://github.com/InternLM/InternLM)
- [LLaMA](https://github.com/facebookresearch/llama)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Baichuan](https://github.com/baichuan-inc)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)
- [ChatGLM3](https://github.com/THUDM/ChatGLM3-6B)
- [TigerBot](https://github.com/TigerResearch/TigerBot)
- [Qwen](https://github.com/QwenLM/Qwen)
- [BlueLM](https://github.com/vivo-ai-lab/BlueLM)
- [Gemma](https://huggingface.co/google/gemma-7b)
- ……

</td>
<td>

- OpenAI
- Gemini
- Claude
- ZhipuAI(ChatGLM)
- Baichuan
- ByteDance(YunQue)
- Huawei(PanGu)
- 360
- Baidu(ERNIEBot)
- MiniMax(ABAB-Chat)
- SenseTime(nova)
- Xunfei(Spark)
- ……

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝返回顶部</a></p>

## 🔜 路线图

- [x] 主观评测
  - [x] 发布主观评测榜单
  - [ ] 发布主观评测数据集
- [x] 长文本
  - [x] 支持广泛的长文本评测集
  - [ ] 发布长文本评测榜单
- [x] 代码能力
  - [ ] 发布代码能力评测榜单
  - [x] 提供非Python语言的评测服务
- [x] 智能体
  - [ ] 支持丰富的智能体方案
  - [x] 提供智能体评测榜单
- [x] 鲁棒性
  - [x] 支持各类攻击方法

## 👷‍♂️ 贡献

我们感谢所有的贡献者为改进和提升 OpenCompass 所作出的努力。请参考[贡献指南](https://opencompass.readthedocs.io/zh_CN/latest/notes/contribution_guide.html)来了解参与项目贡献的相关指引。

## 🤝 致谢

该项目部分的代码引用并修改自 [OpenICL](https://github.com/Shark-NLP/OpenICL)。

该项目部分的数据集和提示词实现修改自 [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub), [instruct-eval](https://github.com/declare-lab/instruct-eval)

## 🖊️ 引用

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

<p align="right"><a href="#top">🔝返回顶部</a></p>
