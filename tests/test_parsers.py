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


def test_unknown_metric_returns_none():
    assert parse_metric("bleu", "bleu: 30") is None


def test_parse_avg_acc_fraction_and_percent():
    assert parse_metric("avg_acc", "avg_acc: 0.6651") == 66.51
    assert parse_metric("avg_acc", "Average accuracy: 66.5%") == 66.5
    # per-task acc lines alone (no avg/average) are not mistaken for the average
    assert parse_metric("avg_acc", "arc_easy: 0.72\npiqa: 0.75") is None


def test_parse_acc_norm_avg_label():
    assert parse_metric("acc_norm_avg", "acc_norm_avg: 0.6651") == 66.51
    assert parse_metric("acc_norm_avg", "ACC_NORM_AVG: 66.5%") == 66.5
