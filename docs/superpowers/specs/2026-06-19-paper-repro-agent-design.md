# 量化论文复现 Agent — 设计文档

> 日期:2026-06-19
> 状态:设计已确认,待写实现计划
> 上游:[llm-paper-radar](https://github.com/zhaolin-amd/llm-paper-radar)(消费其推送的 paper)

## 1. 目标与范围

构建一个 agent,根据 arxiv 上的量化(quantization)论文或其官方 GitHub repo,**复现论文中报告的结果**。论文来源主要是 llm-paper-radar 的推送。

- 有官方 repo 的论文:调用其自带脚本复现(忠实度最高,优先)。
- 无官方 repo 的论文:根据论文描述的算法自己实现(本期**只预留接口,不实现**)。

### 1.1 核心判断

难点不在"跑代码",而在两件事:
1. 把一篇论文翻译成一个**可机器执行、可自动判分的复现规格(spec)**;
2. **诚实地报告差距**——跑不动就说跑不动,绝不用论文数字填空。

跑代码本身在量化领域已高度标准化(PPL + lm-eval-harness + 论文自带脚本),可吃现成生态。

### 1.2 已锁定的设计决定

| 维度 | 决定 |
|---|---|
| 评测 | 优先跑论文自带脚本(official > cited-standard > custom rebuild) |
| Spec | 逐 claim 抽取**完整评测协议** |
| 路径 | 官方 repo 路径先行;从头实现是**预留接口** |
| 判分 | **过程忠实 AND 数值在容差内**,两者都满足才算 MATCH |
| 自主度 | 半自动,门控 1(spec 审批)+ plan 异常哨兵 |
| 运行形态 | 手动 CLI,逐篇调用,状态落文件 |
| 技术栈 | Claude Code headless(setup 调试)+ conda/uv 隔离 |
| 算力 | 多卡可用,成本非约束 |
| 默认 claim | 只抽主结果(论文主推/标黑的几条),其余按需 |
| 默认容差 | PPL ±0.05,accuracy ±0.5%(论文明确给则用论文的) |
| 报告 | 中英双语,md 分两文件 `report.zh.md` / `report.en.md` |

### 1.3 架构选型

采用**确定性流水线,agent 只进 Setup 阶段**(方案 B)。理由:最难的四个约束——忠实+数值的判分、半自动门控、可复现、预留从头实现接口——都指向一个确定性骨架,把 agent 的不确定性关在唯一真正开放式的环节(驯服腐烂的官方 repo 环境)。判分用纯代码做确定性数值比较,不交给 LLM。

## 2. 整体架构

一次 CLI 调用 = 一篇 paper = 一个 run 目录。无队列、无 DB、无 cron。

```
quant-repro run <arxiv_id | .org文件路径 | arxiv_url>
quant-repro resume <run_dir>      # 从上次中断/门控处继续
quant-repro report <run_dir>      # 重新渲染报告
```

七个确定性阶段,Python 编排,每阶段读上一阶段 artifact、写自己的 artifact 到 run 目录:

```
ingest → specextract → plan → setup → run → grade → report
                ⤷[门控1:spec审批]   ⤷[plan:可行性/异常哨兵]
```

| 阶段 | 性质 | 职责 | 产出 |
|---|---|---|---|
| **ingest** | 确定性 | arxiv id → 拉 LaTeX 源码 + 定位官方 repo;入参是 `.org` 则读 `#+source:` | `paper/`、`repo/`、`ingest.json` |
| **specextract** | 1 次 headless 调用 | LaTeX+README → 完整 spec → **停,等审批** | `spec.yaml` |
| **plan** | 确定性 | 估每条 claim 的 GPU/显存/时长 → 可行性/异常检查 | `plan.json` |
| **setup** | agentic 调试循环 | conda/uv 建环境,修依赖直到自带评测命令冒烟通过 | `env/`、`setup_log/`、`env_snapshot.json`、`setup_patches/` |
| **run** | 确定性 | 逐 artifact 量化、逐 claim 调评测脚本,原始输出落盘 | `runs/<claim_id>/` |
| **grade** | 纯代码 | 解析输出,数值+忠实双检,判 MATCH/PARTIAL/FAIL/BLOCKED | `grades.json` |
| **report** | 确定性 | 渲染中英双语报告 | `report.zh.md`、`report.en.md` |

### 2.1 门控

- **门控 1(spec 审批):** specextract 后停,用户过目 `spec.yaml` 再继续。防止抽错协议导致后面全白跑。
- **plan 可行性/异常哨兵:** 默认静默放行(成本非约束)。仅在两种情况升级为一次 `AskUserQuestion`:
  1. **硬件不可行** —— claim 需要环境中根本没有的卡型/显存;
  2. **估算与论文严重背离** —— plan 估算远超论文自报(如论文 4 GPU·时、估出 200),通常意味着 specextract 抽错,是质量信号,值得烧资源前看一眼。

### 2.2 核心隔离原则

grade 是纯代码、与执行分离,只读 run 阶段落盘的原始输出,**永远看不到"该对上的值"之外的执行上下文**。这是"过程忠实+数值双达标"判分能成立、且 agent 无法作弊的前提。

## 3. Ingest 与 Spec Schema

### 3.1 Ingest

入参三形态归一到 arxiv_id:
- radar 的 `.org` 文件 → 读 `#+source:` 拿 arxiv url(radar 已筛过)
- arxiv url / id → 直接用

然后:
- **拉 LaTeX 源码**(`arxiv.org/e-print/<id>`),不 OCR PDF —— 表格数字从 LaTeX 抽准得多。
- **定位官方 repo**,优先级:论文里的 GitHub 链接 > PapersWithCode > GH code search(按标题/方法名)。候选连同置信度写进 `ingest.json`;**找不到则 `repo: null`**(将来走从头实现 provider,本期直接 SKIP 并在报告说明)。

```
ingest.json:
  arxiv_id, title, authors, source_url
  repo: {url, commit, confidence, evidence} | null
  latex_path, repo_path
```

### 3.2 Spec Schema(两层:artifact + claim)

同一量化产物常在多个评测协议下报多条数字,故拆成 **artifacts**(量化产物,可复用)与 **claims**(一条数字 = artifact × 评测协议)。

```yaml
paper: 2401.xxxxx
repo: {url, commit}                    # ingest 带出,grade 记进报告

artifacts:                             # 量化产物
  - id: llama2-7b-w4g128
    base_model: meta-llama/Llama-2-7b-hf
    method: AWQ
    quant_config:                      # 忠实判分要逐项比的就是这些
      wbits: 4
      group_size: 128
      sym: false
      calib: {dataset: pile, n_samples: 128, seqlen: 512}
    calib_status: known                # known | UNKNOWN(抽不到就显式标,grade 判"不可比")

claims:                                # 一条 = 一个判分单元
  - id: c1
    artifact: llama2-7b-w4g128
    eval_protocol:                     # 完整抽取,judge 的依据
      runner: official                 # official | cited-standard | custom
      command: "python eval_ppl.py --model {model} --dataset wikitext2"
      metric: perplexity
      dataset: wikitext2
      split: test
      seqlen: 2048
      stride: 2048
      few_shot: 0
      extra_args: "--use_cache false"
    expected: 5.78
    tolerance: 0.05                    # 默认 PPL ±0.05 / acc ±0.5%;论文给了用论文的
    source: "Table 3, row 2, col W4"   # 可溯源到论文位置
    hardware: null                     # 精度类 null;效率类钉死 "A100-80G,bs=1,seqlen=2048"
```

### 3.3 关键设计点

1. **`runner` 字段是核心**:`official` = 调 repo 自带脚本(优先);`cited-standard` = 论文明确引用的标准实现(如指定版本 lm-eval);`custom` = 都没有,按协议重建,报告标注"非官方实现"。
2. **`calib_status: UNKNOWN` 的显式诚实**:量化复现头号失败原因是 calib 不一致。抽不到就标 UNKNOWN,grade 判"不可比",**绝不默默用默认值蒙**。
3. **`source` 可溯源**:每条 claim 钉到论文 Table/行/列,门控 1 审 spec 时可逐条核对。

### 3.4 SpecExtract(门控 1)

一次 headless 调用,喂 LaTeX 全文 + README,输出上面的 YAML。
- 默认**只抽主结果**(论文主推/标黑的几条),其余按需。
- 容差论文没明说则用默认(PPL ±0.05、acc ±0.5%)并标注"默认值,请确认"。
- 抽完**停**,用户审 `spec.yaml`:核对数字/协议/容差,改完才放行。

## 4. Setup / Run / 失败模式

### 4.1 Setup —— 唯一的 agentic 阶段

唯一交给 Claude Code headless 的环节,因为"驯服腐烂的官方 repo 环境"是唯一真正开放式的问题。

**目标(单一、可判定):** 把 conda/uv 环境修到 **repo 自带评测命令能成功跑通一次**(冒烟测试,非全量)。这是可机器判定的退出条件。

**冒烟测试输入:** (a) repo 自带 example/test 优先;没有则 (b) fallback 到 spec 某条 claim 命令缩到极小规模(如 8 样本、1 batch)。

**循环:**
```
建环境(conda/uv) → 装依赖 → 冒烟跑评测命令
   → 失败 → agent 读 traceback → 改(版本钉/补包/改API) → 重试
   → 成功 → 冻结环境快照 → 退出
```

**护栏:**
- **重试上限 + 总超时**:超了不静默放弃,停下标 `setup: FAILED`,完整 setup_log 交用户介入。
- **环境快照入库**:成功后 `pip freeze` + CUDA/torch/transformers 版本写进 `env_snapshot.json`。这是官方 repo 路径头号假阴性来源(依赖漂移),报告必记。
- **agent 改动留痕**:每个 patch(改了哪行 API、钉了哪个版本)记进 `setup_patches/`,可能正是复现失败的原因,grade 与报告要能看到。

**为什么 setup 与 run 分离:** setup 只负责"环境能跑起来",不碰真实验参数。agent 的不确定性被关在"让它能跑"这一步,不渗进"跑出什么数"。

### 4.2 Run —— 确定性执行

setup 通过后无 agent。按 spec 逐条执行:
```
for artifact in spec.artifacts:
    按 quant_config 量化(调 repo 量化入口或提供的 checkpoint)
for claim in spec.claims:
    按 eval_protocol.command 跑评测,原始 stdout/产物落盘 runs/<claim_id>/
    记录:实际命令、seed、起止时间、用的 GPU
```

**优先用官方复现命令**:很多 repo 直接提供量化后 checkpoint 或一条 `python main.py --reproduce`。run 优先用它,而非自己拼参数——忠实度最高、最不易踩坑。此时 quant_config 退化为判分依据(grade 用它核对忠实度),非执行指令。

**run 不解析结果、不判分**,只忠实落盘原始输出。

### 4.3 量化领域特有失败模式(提前埋检查)

| 失败模式 | 在哪埋检查 |
|---|---|
| calib 不一致(split/条数/seqlen) | specextract 抽 calib;抽不到标 UNKNOWN → grade 判"不可比" |
| PPL 口径(stride、seqlen) | eval_protocol 必抽 seqlen/stride;grade 核对 |
| 官方 repo 依赖腐烂 | setup 调试循环 + 环境快照 |
| 复现了精度但卖点是 speedup | 报告分开讲"复现了哪一半";效率类 claim 单独标 hardware |
| agent 改 repo 导致数字偏移 | setup_patches 留痕,报告暴露 |

## 5. Grade 判分 + Report

### 5.1 Grade —— 纯代码,与执行隔离

只读 run 落盘的原始输出 + spec,**不重跑、看不到"该对上的值"之外的执行上下文**。

每条 claim 两道独立检查,都过才 MATCH:

**检查 1 · 数值达标**
```
解析 runs/<claim_id>/ → measured
pass_value = |measured - expected| <= tolerance
解析不出 → UNPARSEABLE(不猜)
```
解析器按 metric 类型写(PPL、accuracy、speedup…),从已知评测脚本输出格式提取。

**检查 2 · 过程忠实**
```
逐项比对 实际 config vs spec.eval_protocol / quant_config:
  seqlen, stride, calib, wbits, group_size, few_shot...
pass_faithful = 全部关键项一致
calib_status==UNKNOWN → 判不可比
setup_patches 含影响数值的改动 → 标记降级
```

**三态 + BLOCKED:**

| 判定 | 条件 |
|---|---|
| **MATCH** | 数值达标 AND 过程忠实 |
| **PARTIAL** | 数值达标但过程有偏差;或过程忠实但数值超容差(必带原因) |
| **FAIL** | 数值显著偏离且无法归因 |
| **BLOCKED** | setup 失败 / 输出无法解析 / 评测没跑成 —— "没跑成"不是"复现失败",单独成态 |

PARTIAL 一定带原因(哪项 config 不一致、偏多少)。绝不"接近就算过"。BLOCKED 与 FAIL 分开:"环境没搭起来"和"方法没复现"是两码事。

### 5.2 Report —— 确定性渲染,中英双语两文件

每篇产出 `report.zh.md` 与 `report.en.md`,核心是一张可溯源、可复算的表:

```markdown
# 复现报告:<title> (<arxiv_id>)
repo: <url>@<commit> | 环境: torch X / transformers Y / CUDA Z
判定汇总: MATCH 3 / PARTIAL 1 / FAIL 0 / BLOCKED 1

| claim | 模型 | 配置 | 指标 | paper | 实测 | 判定 | 原因 |
|-------|------|------|------|-------|------|------|------|
| c1 | Llama2-7B | W4G128 | wiki2 PPL | 5.78 | 5.80 | MATCH | — |
| c2 | Llama2-7B | W3G128 | wiki2 PPL | 6.92 | 7.41 | PARTIAL | 超容差 0.49;calib n_samples 抽不到用默认 128 |
| c3 | ... | | speedup | 2.1x | — | BLOCKED | setup 失败:cuda kernel 编译不过 |

## 复算信息(每条 claim)
c1: 命令 `...` | seed 0 | GPU A100×1 | 用时 18min | 原始输出 runs/c1/stdout.log
## Setup 改动留痕
- 钉 transformers==4.36(repo 要求 4.31 但与 torch 2.x 冲突)
## 复现了哪一半
精度类 3/4 复现;效率类 1 条 BLOCKED —— paper 主卖点含 speedup,该部分未验证
```

**报告铁律:**
- 永远是实测原始数字,绝不用论文数字填空。
- 每条 claim 附完整复算命令 + seed + commit + 环境,任何人可照着重跑。
- "复现了哪一半"显式写 —— 精度复现 ≠ speedup 复现。
- judge 逻辑与执行分离,判定可追溯到 grade 的两道检查。

### 5.3 沉淀(轻量)

每篇复现产出可复用资产,落目录而非数据库:
- `env_snapshot.json` + `setup_patches` → 下次同方法/同 repo 系直接复用。
- method adapter(AWQ/GPTQ 系)→ 攒成 `adapters/`,遇变体复用基座。

CLI 形态下这是"约定的目录布局",不是额外基础设施。

## 6. 从头实现路径(本期预留接口)

无官方 repo 的论文走"从头实现"。本期不实现,但架构预留:
- **Provider 接口**:官方 repo 路径是一个 `OfficialRepoProvider`,从头实现是另一个 `FromScratchProvider`,二者实现同一接口(产出量化产物 + 可执行评测命令),汇流到同一 grade/report。
- ingest 标 `repo: null` 的论文,本期直接 SKIP 并在报告说明"无官方 repo,待从头实现"。

## 7. Run 目录布局

```
runs/<arxiv_id>-<timestamp>/
  ingest.json
  paper/                 # LaTeX 源码
  repo/                  # clone 的官方 repo
  spec.yaml
  plan.json
  env/                   # conda/uv 环境(或引用)
  env_snapshot.json
  setup_log/
  setup_patches/
  runs/<claim_id>/       # 每条 claim 的原始输出、命令、seed
  grades.json
  report.zh.md
  report.en.md
```

## 8. 不做什么(YAGNI)

- 不做 job 队列 / DB / cron(手动 CLI,文件状态即可)。
- 不做跨论文调度器(多卡只服务单篇 run 内的 claim)。
- 不做成本预算审批门控(资源非约束)。
- 不做 LLM 判分(数值比较用代码)。
- 本期不实现从头实现 provider(只留接口)。
