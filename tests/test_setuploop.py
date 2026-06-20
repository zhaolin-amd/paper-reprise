from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.rundir import RunDir
from paper_reprise.setuploop import select_smoke_command, shrink_command


def _spec(command="python eval_ppl.py --model m --dataset wikitext2"):
    return Spec(
        paper="p", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ",
                            quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="official", command=command,
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_shrink_command_appends_tiny_scale_flags():
    out = shrink_command("python eval_ppl.py --model m")
    assert out == "python eval_ppl.py --model m --limit 8 --batch-size 1"


def test_select_smoke_prefers_repo_example(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.repo_dir / "examples").mkdir()
    (rd.repo_dir / "examples" / "smoke.sh").write_text("echo hi")
    cmd = select_smoke_command(rd, _spec())
    assert cmd == "bash examples/smoke.sh"


def test_select_smoke_falls_back_to_shrunk_claim_command(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")  # empty repo, no example
    cmd = select_smoke_command(rd, _spec())
    assert cmd == "python eval_ppl.py --model m --dataset wikitext2 --limit 8 --batch-size 1"
