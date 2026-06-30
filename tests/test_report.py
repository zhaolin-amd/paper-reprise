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
    assert "?" not in zh                # no "?" anywhere from unknown env
    assert "环境" not in zh              # the env bullet is omitted entirely when nothing known


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


def test_header_is_a_markdown_meta_block_not_glued_lines():
    spec, ingest, grades, runs, env = _ctx()
    zh, en = render_reports(spec, ingest, grades, runs, env, patches=[])
    # repo / env / verdict are separate bullet lines, not one run-on paragraph
    assert "- **Repo:** https://github.com/x/y@abc123" in en
    assert "- **Verdict:** MATCH 1 · PARTIAL 0 · FAIL 0 · BLOCKED 0" in en
    assert "- **仓库:**" in zh and "- **判定:**" in zh
    # blank line after the H1 so Markdown doesn't fold the metadata into the title
    assert en.splitlines()[1] == ""


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
