"""Tests for the run-diagnostics breadcrumbs."""
import json
import diagnostics


def test_cuda_status_has_keys():
    s = diagnostics.cuda_status()
    assert "cuda" in s and "device" in s
    assert s["cuda"] in (True, False, None)


def test_run_context_emits_and_returns(capsys):
    ctx = diagnostics.run_context(stage="unit_test", checkpoint="x.pt",
                                  dataset="data/foo", output="out/bar")
    # returned dict carries the fields + standard breadcrumbs
    assert ctx["stage"] == "unit_test"
    assert ctx["checkpoint"] == "x.pt"
    assert "ts" in ctx and "device" in ctx
    # one parseable [runctx] line was printed
    line = capsys.readouterr().out.strip()
    assert line.startswith("[runctx] ")
    parsed = json.loads(line[len("[runctx] "):])
    assert parsed["dataset"] == "data/foo"


def test_run_failure_breadcrumb(capsys):
    rec = diagnostics.run_failure("cnn_train", ValueError("boom"))
    assert rec["stage"] == "cnn_train"
    assert rec["error_type"] == "ValueError"
    out = capsys.readouterr().out
    assert "[runfail]" in out and "boom" in out
