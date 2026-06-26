from paper_reprise.parsers import parse_metric


def test_parse_perplexity_simple():
    out = "Evaluating...\nwikitext2 perplexity: 5.80\nDone."
    assert parse_metric("perplexity", out) == 5.80


def test_parse_perplexity_ppl_alias():
    out = "final PPL = 7.41"
    assert parse_metric("perplexity", out) == 7.41


def test_parse_accuracy_percent():
    out = "hellaswag acc: 76.3%"
    assert parse_metric("accuracy", out) == 76.3


def test_parse_accuracy_fraction_normalized_to_percent():
    out = "acc,none: 0.763"
    assert parse_metric("accuracy", out) == 76.3


def test_parse_accuracy_lm_eval_markdown_table():
    # lm-eval prints results as a markdown table; the `acc` row value (not acc_norm)
    # must be picked and normalized to a percentage.
    out = (
        "|  Tasks |Version|Filter|n-shot| Metric |   |Value|   |Stderr|\n"
        "|--------|------:|------|-----:|--------|---|----:|---|-----:|\n"
        "|arc_easy|      1|none  |     0|acc     |↑  |0.500|±  | 0.189|\n"
        "|        |       |none  |     0|acc_norm|↑  |0.375|±  | 0.183|\n"
    )
    assert parse_metric("acc", out) == 50.0


def test_unparseable_returns_none():
    assert parse_metric("perplexity", "garbage with no number relevant") is None


def test_unknown_metric_falls_back_to_generic_scalar_line():
    # A metric outside the known families (ppl/acc/avg/speedup) is read verbatim from a
    # standalone `<metric>: <number>` line — the from-scratch path's scalar metrics.
    assert parse_metric("bleu", "bleu: 30") == 30.0


def test_generic_scalar_not_percent_normalized():
    # Distortion is genuinely sub-1; it must NOT be x100'd the way acc/avg fractions are.
    assert parse_metric("mse_distortion", "mse_distortion: 0.36") == 0.36
    assert parse_metric("ip_distortion", "ip_distortion: 3.06e-5") == 3.06e-5


def test_generic_scalar_requires_standalone_line():
    # Anchored to a whole line: prose / tracebacks mentioning the name don't count.
    assert parse_metric("ip_bias", "computing ip_bias: takes 0 retries here") is None
    assert parse_metric("recall", "see recall at file.py:200 for details") is None
    assert parse_metric("recall", "nothing relevant") is None


def test_generic_scalar_signed_value():
    assert parse_metric("ip_bias", "ip_bias: -0.0004") == -0.0004


def test_parse_avg_acc_fraction_and_percent():
    assert parse_metric("avg_acc", "avg_acc: 0.6651") == 66.51
    assert parse_metric("avg_acc", "Average accuracy: 66.5%") == 66.5
    # per-task acc lines alone (no avg/average) are not mistaken for the average
    assert parse_metric("avg_acc", "arc_easy: 0.72\npiqa: 0.75") is None


def test_parse_acc_norm_avg_label():
    assert parse_metric("acc_norm_avg", "acc_norm_avg: 0.6651") == 66.51
    assert parse_metric("acc_norm_avg", "ACC_NORM_AVG: 66.5%") == 66.5


def test_parse_per_task_extracts_tasks_skips_average():
    from paper_reprise.parsers import parse_per_task
    txt = ("# header line\narc_challenge acc_norm: 0.459\narc_easy: 0.7285\n"
           "winogrande acc: 0.6922\nacc_norm_avg: 0.6651\n")
    d = parse_per_task(txt)
    assert d == {"arc_challenge": 0.459, "arc_easy": 0.7285, "winogrande": 0.6922}
