from paper_reprise.models import Artifact, Claim, EvalProtocol, Spec
from paper_reprise.provider import (
    make_run_dispatcher,
    make_setup_dispatcher,
    repo_present,
)
from paper_reprise.rundir import RunDir


def _spec():
    return Spec(
        paper="2401.00001", repo=None,
        artifacts=[Artifact(id="a1", base_model="m", method="AWQ", quant_config={"wbits": 4})],
        claims=[Claim(id="c1", artifact="a1",
                      eval_protocol=EvalProtocol(runner="custom", command="x",
                                                 metric="perplexity", dataset="wikitext2"),
                      expected=5.78, tolerance=0.05, source="T")],
    )


def test_repo_present_false_when_repo_dir_empty(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")   # repo_dir mkdir-ed, empty
    assert repo_present(rd) is False


def test_repo_present_true_when_repo_dir_has_content(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.repo_dir / "README.md").write_text("cloned repo")
    assert repo_present(rd) is True


def test_setup_dispatcher_routes_to_official_when_repo_present(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    (rd.repo_dir / "setup.py").write_text("x")
    called = {}
    dispatcher = make_setup_dispatcher(
        official=lambda rd, spec: called.setdefault("which", "official"),
        fromscratch=lambda rd, spec: called.setdefault("which", "fromscratch"))
    dispatcher(rd, _spec())
    assert called["which"] == "official"


def test_setup_dispatcher_routes_to_fromscratch_when_no_repo(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")   # empty repo_dir
    called = {}
    dispatcher = make_setup_dispatcher(
        official=lambda rd, spec: called.setdefault("which", "official"),
        fromscratch=lambda rd, spec: called.setdefault("which", "fromscratch"))
    dispatcher(rd, _spec())
    assert called["which"] == "fromscratch"


def test_run_dispatcher_routes_by_repo_presence(tmp_path):
    rd = RunDir.create(tmp_path, arxiv_id="p", timestamp="t")
    claim_dir = rd.claim_dir("c1")
    called = {}
    dispatcher = make_run_dispatcher(
        official=lambda claim, art, cd: called.setdefault("which", "official"),
        fromscratch=lambda claim, art, cd: called.setdefault("which", "fromscratch"))
    # no repo content → from-scratch
    dispatcher(_spec().claims[0], _spec().artifacts[0], claim_dir)
    assert called["which"] == "fromscratch"
    # now add a repo → official
    (rd.repo_dir / "main.py").write_text("x")
    called.clear()
    dispatcher(_spec().claims[0], _spec().artifacts[0], claim_dir)
    assert called["which"] == "official"
