# paper-reprise

复现量化(quantization)论文结果的 agent。输入一篇 arxiv 论文(或 llm-paper-radar 推送的 `.org`),
优先调用其官方 repo 的自带脚本复现论文报告的数字,诚实地报告差距。

设计文档:[docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md](docs/superpowers/specs/2026-06-19-paper-reprise-agent-design.md)

## 状态

设计已确认,待写实现计划。

## 形态(规划中)

```
paper-reprise run <arxiv_id | .org文件路径 | arxiv_url>
paper-reprise resume <run_dir>
paper-reprise report <run_dir>
```

确定性流水线 `ingest → specextract → plan → setup → run → grade → report`,
agent 只进 setup 阶段(驯服腐烂的官方 repo 环境),判分用纯代码、与执行隔离。
