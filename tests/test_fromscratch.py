from paper_reprise.fromscratch import (
    build_scaffold_prompt,
    fromscratch_eval_command,
    fromscratch_smoke_command,
)
from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir


def _spec(command="python eval_ppl.py --model m --dataset wikitext2"):
    return Spec(
        paper="2401.00001", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                            quant_config={"wbits": 4, "group_size": 128})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="custom", command=command,
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="Table 3")],
    )


def test_smoke_command_is_tiny_scale_entrypoint():
    assert fromscratch_smoke_command() == "bash impl/run_eval.sh --smoke"


def test_eval_command_invokes_entrypoint_with_claim_id():
    assert fromscratch_eval_command(_spec().claims[0]) == "bash impl/run_eval.sh c1"


def test_scaffold_prompt_instructs_impl_entrypoint_and_forbids_fabrication(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="2401.00001", timestamp="t")
    prompt = build_scaffold_prompt(rd, _spec())
    # points at the paper source + the spec
    assert "paper/" in prompt
    assert "spec.yaml" in prompt
    # the single runnable entrypoint contract
    assert "impl/run_eval.sh" in prompt
    # the method to implement is surfaced from the spec
    assert "AWQ" in prompt
    # honesty rule: must NOT fabricate numbers
    low = prompt.lower()
    assert "fabricat" in low or "do not invent" in low or "must not invent" in low
    # patch-note discipline: one-line note per file
    assert "one line" in low or "one-line" in low
