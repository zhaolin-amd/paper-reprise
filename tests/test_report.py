from paper_reprise.models import (
    Spec, Artifact, Claim, EvalProtocol, RepoInfo, ClaimGrade, RunResult, IngestInfo,
)
from paper_reprise.report import render_reports


def _ctx():
    spec = Spec(
        paper="2401.00001",
        repo=RepoInfo(url="https://github.com/x/y", commit="abc123"),
        artifacts=[Artifact(id="a1", base_model="Llama2-7B", method="AWQ",
                            quant_config={"wbits": 4, "group_size": 128})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command="python e.py",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="Table 3")],
    )
    ingest = IngestInfo(arxiv_id="2401.00001", title="Test Paper",
                        source_url="https://arxiv.org/abs/2401.00001",
                        repo=RepoInfo(url="https://github.com/x/y", commit="abc123"))
    grades = [ClaimGrade(claim_id="c1", verdict="MATCH", measured=5.80, expected=5.78,
                         reason="—", checks={"value": True, "faithful": True})]
    runs = [RunResult(claim_id="c1", command="python e.py", seed=0, gpu="A100x1",
                      minutes=18.0, stdout_path="claims/c1/stdout.log")]
    env = {"torch": "2.3.0", "transformers": "4.36.0", "cuda": "12.1"}
    return spec, ingest, grades, runs, env


def test_renders_both_languages():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "复现报告" in zh
    assert "Reproduction Report" in en


def test_uses_measured_not_expected_for_actual_column():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "5.80" in zh
    assert "MATCH" in zh


def test_measured_column_annotates_diff_vs_paper():
    spec, ingest, grades, runs, env = _ctx()  # measured 5.80, expected 5.78
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    for doc in (zh, en):
        assert "5.80(+0.02)" in doc


def test_sub_one_metric_keeps_significant_figures():
    # A distortion-style metric (e.g. TurboQuant) is genuinely sub-1; the measured cell
    # must keep significant figures, not collapse to 0.00 the way `.2f` would.
    spec, ingest, grades, runs, env = _ctx()
    spec.artifacts[0].method = "TurboQuant_prod"
    spec.claims[0].eval_protocol.metric = "ip_distortion"
    spec.claims[0].expected = 3.0599e-5
    grades[0].measured = 3.5449e-5
    grades[0].expected = 3.0599e-5
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    for doc in (zh, en):
        assert "3.54e-05" in doc           # measured shown with sig figs
        assert "0.00(+0.00)" not in doc    # NOT crushed by .2f


def test_summary_table_columns_model_config_algorithm():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    for doc in (zh, en):
        assert "| config | algorithm | metric |" in doc       # split columns present
        assert "| Llama2-7B | INT4 G128 | AWQ |" in doc        # config + algorithm derived


def test_baseline_artifact_bf16_algorithm_is_dash():
    from paper_reprise.report import _config_label, _algorithm_label
    from paper_reprise.models import Artifact
    base = Artifact(id="b", base_model="m", method="none", quant_config={"wbits": 16})
    assert _config_label(base) == "BF16"
    assert _algorithm_label(base) == "-"  # uncompressed -> no algorithm


def test_single_table_only_no_separate_summary_and_detail():
    # two metrics -> two rows in ONE table (no separate 汇总/明细 tables)
    art = Artifact(id="a1", base_model="Llama-8B", method="GSQ",
                   quant_config={"wbits": 2, "group_size": 128})
    def ep(metric):
        return EvalProtocol(runner="official", command="x", metric=metric, dataset="d")
    spec = Spec(paper="p", repo=None, artifacts=[art], claims=[
        Claim(id="c_mmlu", artifact="a1", eval_protocol=ep("mmlu"),
              expected=60.0, tolerance=0.5, source="T"),
        Claim(id="c_gsm", artifact="a1", eval_protocol=ep("gsm8k"),
              expected=44.1, tolerance=0.5, source="T"),
    ])
    grades = [
        ClaimGrade(claim_id="c_mmlu", verdict="PARTIAL", measured=58.0, expected=60.0,
                   reason="off", checks={"value": False, "faithful": True}),
        ClaimGrade(claim_id="c_gsm", verdict="FAIL", measured=41.0, expected=44.1,
                   reason="off", checks={"value": False, "faithful": True}),
    ]
    zh, en = render_reports(spec, IngestInfo(arxiv_id="p", source_url="u"),
                            grades, [], env={}, patches=[])
    for doc in (zh, en):
        assert doc.count("| Llama-8B | INT2 G128 | GSQ |") == 2   # one row per metric
        assert "mmlu" in doc and "gsm8k" in doc
    # no second table section
    assert "明细" not in zh and "Details" not in en


def test_patches_section_omitted_when_empty_shown_when_present():
    spec, ingest, grades, runs, env = _ctx()
    zh_empty, en_empty = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "改动留痕" not in zh_empty and "Setup patches" not in en_empty
    zh, en = render_reports(spec, ingest, grades, runs, env,
                            patches=["bumped numpy pin to build"])
    assert "改动留痕" in zh and "bumped numpy pin to build" in zh
    assert "Setup patches" in en and "bumped numpy pin to build" in en


def test_replay_grouped_per_config_dedupes_shared_command():
    art = Artifact(id="a1", base_model="Llama-8B", method="GSQ",
                   quant_config={"wbits": 2, "group_size": 128})
    def ep(metric):
        return EvalProtocol(runner="official", command="serve && eval", metric=metric,
                            dataset="d")
    spec = Spec(paper="p", repo=None, artifacts=[art], claims=[
        Claim(id="c_mmlu", artifact="a1", eval_protocol=ep("mmlu"),
              expected=60.0, tolerance=0.5, source="T"),
        Claim(id="c_gsm", artifact="a1", eval_protocol=ep("gsm8k"),
              expected=44.1, tolerance=0.5, source="T"),
    ])
    runs = [RunResult(claim_id="c_mmlu", command="serve && eval", stdout_path="a/x.log"),
            RunResult(claim_id="c_gsm", command="serve && eval", stdout_path="b/y.log")]
    zh, en = render_reports(spec, IngestInfo(arxiv_id="p", source_url="u"),
                            [], runs, env={}, patches=[])
    for doc in (zh, en):
        assert "Replay script (per config)" in en
        assert "复算脚本(每个 config)" in zh
        # one config heading, shared command shown once (deduped), both stdouts listed
        assert "**Llama-8B · INT2 G128 · GSQ**" in doc
        assert doc.count("serve && eval") == 1
        assert "a/x.log" in doc and "b/y.log" in doc


def test_summary_counts_verdicts():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "MATCH 1" in zh


def test_env_snapshot_in_report():
    spec, ingest, grades, runs, env = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "4.36.0" in zh
    assert "abc123" in zh


def test_repo_falls_back_to_spec_when_ingest_repo_missing():
    spec, ingest, grades, runs, env = _ctx()
    ingest.repo = None  # ingest didn't capture it, but spec.repo did
    zh, _ = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert "(no official repo)" not in zh
    assert "github.com/x/y" in zh


def test_unknown_env_values_are_dropped_not_question_marked():
    spec, ingest, grades, runs, _ = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs,
                           env={"torch": "unknown", "transformers": "", "cuda": None},
                           patches=[])
    assert "- **环境:**" not in zh       # the env bullet is omitted when nothing known
    assert "torch" not in zh             # no torch version anywhere in the report


def test_known_env_values_are_kept_unknown_ones_dropped():
    spec, ingest, grades, runs, _ = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs,
                           env={"torch": "2.3.0", "transformers": "", "cuda": None},
                           patches=[])
    assert "**环境:** torch 2.3.0" in zh
    assert "transformers" not in zh and "CUDA" not in zh


def test_rocm_version_is_printed_for_amd_builds():
    spec, ingest, grades, runs, _ = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs,
                           env={"torch": "2.3.0", "transformers": "4.40.0",
                                "cuda": "unknown", "rocm": "6.1"},
                           patches=[])
    assert "ROCm 6.1" in zh
    assert "CUDA" not in zh             # AMD build: CUDA omitted, ROCm shown


def test_lm_eval_version_shown_in_env():
    spec, ingest, grades, runs, _ = _ctx()
    zh, _ = render_reports(spec, ingest, grades, runs,
                           env={"torch": "2.3.0", "lm_eval": "0.4.12"}, patches=[])
    assert "lm_eval 0.4.12" in zh


def test_versions_recovered_from_eval_logs(tmp_path):
    # The eval prints its own versions; recover them when the snapshot missed them
    # (e.g. eval ran in a shared env, or pip freeze came back empty).
    from paper_reprise.models import RunResult
    from paper_reprise.report import _effective_env, _versions_from_logs

    log = tmp_path / "stdout.log"
    log.write_text("Torch        : 2.10.0+rocm7.1\nTransformers : 5.13.0\n"
                   "\x1b[38;20mINFO Using lm-eval version 0.4.12\x1b[0m\n")
    r = RunResult(claim_id="c", command="x", stdout_path=str(log),
                  status="ran", block_reason=None)
    assert _versions_from_logs([r]) == {
        "torch": "2.10.0+rocm7.1", "transformers": "5.13.0", "lm_eval": "0.4.12"}
    # unknown snapshot fields get filled from logs; known ones are kept
    eff = _effective_env({"torch": "unknown", "cuda": "13.0"}, [r])
    assert eff["torch"] == "2.10.0+rocm7.1"
    assert eff["lm_eval"] == "0.4.12"
    assert eff["cuda"] == "13.0"


def test_header_is_a_markdown_meta_block_not_glued_lines():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    # repo / env are separate bullet lines, not one run-on paragraph
    assert "- **Repo:** https://github.com/x/y@abc123" in en
    assert "- **仓库:**" in zh
    # blank line after the H1 so Markdown doesn't fold the metadata into the title
    assert en.splitlines()[1] == ""


def test_verdict_counts_only_in_conclusion_not_header():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    # the verdict counts are no longer duplicated in the header meta block
    assert "- **Verdict:**" not in en and "- **判定:**" not in zh
    # they appear once, in the Conclusion section
    assert "MATCH 1 · PARTIAL 0 · FAIL 0 · BLOCKED 0" in en
    assert en.index("## Conclusion") < en.index("MATCH 1 ·")


def test_title_not_duplicated_when_no_distinct_title():
    spec, ingest, grades, runs, env = _ctx()
    ingest.title = None                 # no title fetched -> show the id once, not "(id id)"
    _, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    assert en.splitlines()[0] == "# Reproduction Report: 2401.00001"


def test_report_includes_per_task_raw_scores(tmp_path):
    from paper_reprise.report import render_reports
    from paper_reprise.models import (Spec, Artifact, Claim, EvalProtocol,
                                       IngestInfo, ClaimGrade, RunResult)
    art = Artifact(id="a1", base_model="m", method="GSQ", quant_config={"wbits": 2})
    claim = Claim(id="c1", artifact="a1",
                  eval_protocol=EvalProtocol(runner="official", command="x",
                                             metric="acc_norm_avg", dataset="arc_easy,piqa"),
                  expected=68.55, tolerance=0.5, source="T")
    spec = Spec(paper="p", repo=None, artifacts=[art], claims=[claim])
    log = tmp_path / "stdout.log"
    log.write_text("|Tasks|Metric|Value|\n|---|---|---|\n|arc_easy|acc_norm|0.73|\n|piqa|acc|0.76|\nacc_norm_avg: 0.745\n")
    run = RunResult(claim_id="c1", command="x", stdout_path=str(log), status="ran")
    grade = ClaimGrade(claim_id="c1", verdict="PARTIAL", measured=74.5, expected=68.55,
                       reason="-", checks={"value": False, "faithful": True})
    zh, en = render_reports(spec, IngestInfo(arxiv_id="p", source_url="u"),
                            [grade], [run], env={}, patches=[])
    for doc in (zh, en):
        assert "|arc_easy|acc_norm|0.73|" in doc      # raw table embedded verbatim
        assert "|piqa|acc|0.76|" in doc
    assert "各任务原始分数" in zh and "Per-task raw scores" in en


def test_raw_scores_drops_mmlu_subgroup_rows(tmp_path):
    from paper_reprise.report import _raw_scores
    from paper_reprise.models import RunResult
    log = tmp_path / "stdout.log"
    log.write_text(
        "|   Tasks   |Version|Filter|n-shot|Metric|   |Value |\n"
        "|-----------|------:|------|-----:|------|---|-----:|\n"
        "|mmlu       |      2|none  |      |acc   |   |0.5000|\n"
        "| - humanities|    2|none  |     0|acc   |↑  |0.5200|\n"
        "|  - formal_logic|1|none |     0|acc   |↑  |0.3900|\n"
        "|winogrande |      1|none  |     0|acc   |↑  |0.7000|\n"
    )
    out = _raw_scores([RunResult(claim_id="c1", command="x",
                                 stdout_path=str(log), status="ran")])
    assert "mmlu" in out and "winogrande" in out      # top-level tasks kept
    assert "humanities" not in out                     # MMLU sub-items dropped
    assert "formal_logic" not in out
    assert "|------" in out or "-----:" in out         # separator preserved


def test_conclusion_reports_counts_and_consistent_offset():
    from paper_reprise.report import _conclusion
    spec, _ing, grades, _runs, _env = _ctx()
    grades[0].measured, grades[0].expected, grades[0].verdict = 6.0, 5.0, "PARTIAL"
    zh = _conclusion(spec, grades, "zh")
    en = _conclusion(spec, grades, "en")
    assert "PARTIAL 1" in zh and "PARTIAL 1" in en
    assert "偏高" in zh and "consistently above" in en   # all diffs same (positive) sign


def test_resources_table_has_time_and_vram(tmp_path):
    from paper_reprise.report import _resources
    from paper_reprise.models import RunResult
    log = tmp_path / "s.log"
    log.write_text("2026-06-30 09:00:00 x\n'peak_vram': 12.5GB\n2026-06-30 09:05:00 y\n")
    spec, _ing, _g, _r, _e = _ctx()
    runs = [RunResult(claim_id="c1", command="x", stdout_path=str(log), status="ran")]
    out = _resources(spec, runs, "| model | config | time | peak VRAM |")
    assert "5.0 min" in out and "12.5 GB" in out


def _has_cjk(s):
    return any('一' <= ch <= '鿿' for ch in s)


def test_en_report_reason_is_english_zh_is_chinese():
    from paper_reprise.models import ClaimGrade
    spec, ingest, _g, runs, env = _ctx()
    grades = [ClaimGrade(claim_id="c1", verdict="PARTIAL", measured=6.0, expected=5.78,
                         reason="process faithful but value off tolerance 0.22 (>0.05)",
                         reason_zh="过程忠实但数值超容差 0.22 (>0.05)",
                         checks={"value": False, "faithful": True})]
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    en_table = en.split("## Conclusion")[0]
    assert not _has_cjk(en_table)                      # the en verdict table has no Chinese
    assert "process faithful but value off tolerance" in en
    assert "过程忠实但数值超容差" in zh                  # zh keeps Chinese reason


def test_conclusion_baseline_validates_so_quant_gap_is_real():
    # FP baseline MATCH + a quantized config off-tolerance → conclusion says the eval
    # protocol is validated and the quantized gap is a real reproduction gap.
    from paper_reprise.models import Spec, Artifact, Claim, EvalProtocol, ClaimGrade
    from paper_reprise.report import _conclusion
    spec = Spec(paper="p",
        artifacts=[Artifact(id="fp", base_model="M", method="none", quant_config={"wbits": 16}),
                   Artifact(id="q", base_model="M", method="GSQ", quant_config={"wbits": 2})],
        claims=[Claim(id="base", artifact="fp",
                      eval_protocol=EvalProtocol(runner="custom", command="x",
                                                 metric="avg_acc", dataset="d"),
                      expected=73.71, tolerance=0.5, source="T"),
                Claim(id="q2", artifact="q",
                      eval_protocol=EvalProtocol(runner="custom", command="x",
                                                 metric="avg_acc", dataset="d"),
                      expected=68.55, tolerance=0.5, source="T")])
    grades = [
        ClaimGrade(claim_id="base", verdict="MATCH", measured=73.79, expected=73.71,
                   reason="—", checks={"value": True, "faithful": True}),
        ClaimGrade(claim_id="q2", verdict="PARTIAL", measured=66.51, expected=68.55,
                   reason="...", checks={"value": False, "faithful": True}),
    ]
    zh = _conclusion(spec, grades, "zh")
    en = _conclusion(spec, grades, "en")
    assert "评测协议可信" in zh and "真实的复现差距" in zh
    assert "eval protocol is validated" in en and "genuine reproduction gap" in en
    assert "-2.04" in en or "-2.04" in zh        # worst quantized gap surfaced


def test_env_order_is_hardware_to_eval():
    from paper_reprise.report import _env_str
    s = _env_str({"torch": "2.1", "transformers": "4.4", "lm_eval": "0.4", "cuda": "13.0"})
    assert s == "CUDA 13.0 / torch 2.1 / transformers 4.4 / lm_eval 0.4"
    # AMD: ROCm takes the accelerator slot, still first
    s2 = _env_str({"torch": "2.1", "rocm": "6.1"})
    assert s2 == "ROCm 6.1 / torch 2.1"


def test_analysis_section_appended_after_conclusion(tmp_path):
    from paper_reprise.report import render_reports
    spec, ingest, grades, runs, env = _ctx()
    analysis = "The gap is due to inference engine mismatch (vLLM vs HF)."
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[], analysis=analysis)
    # analysis appears after Conclusion, before Resources
    assert "## Analysis\n" + analysis in en
    assert "## 差距分析\n" + analysis in zh
    assert en.index("## Analysis") > en.index("## Conclusion")
    assert en.index("## Analysis") < en.index("## Resources")


def test_analysis_section_omitted_when_empty(tmp_path):
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[], analysis="")
    assert "## Analysis" not in en
    assert "## 差距分析" not in zh
