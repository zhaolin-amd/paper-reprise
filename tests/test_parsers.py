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


def test_parse_speedup():
    out = "Throughput speedup: 2.1x over fp16"
    assert parse_metric("speedup", out) == 2.1


def test_unparseable_returns_none():
    assert parse_metric("perplexity", "garbage with no number relevant") is None


def test_unknown_metric_returns_none():
    assert parse_metric("bleu", "bleu: 30") is None
