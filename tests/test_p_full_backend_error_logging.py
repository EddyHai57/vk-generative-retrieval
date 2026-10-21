"""Error diagnostics retain the existing machine-readable stdout receipt."""
import json


def test_plan_error_message_goes_to_stderr_without_stdout_schema_change(monkeypatch,capsys):
    from analysis import p_full_training_backend as cli
    def fail(*args):raise ValueError('synthetic transport boundary')
    monkeypatch.setattr(cli,'plan',fail)
    assert cli.main(['plan','--stage','preflight','--inputs','synthetic-input','--request','synthetic-request','--request-sha256','0'*64])==2
    result=capsys.readouterr()
    assert json.loads(result.out)=={'status':'REJECTED','data_opened':'UNKNOWN','training':False,'verified':False,'error_type':'ValueError'}
    assert json.loads(result.err)=={'stage':'entry_failed','error_type':'ValueError','message':'synthetic transport boundary'}
