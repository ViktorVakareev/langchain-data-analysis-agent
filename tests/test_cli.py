from data_agent.cli import main


def test_cli_answer(capsys, data_dir):
    assert main(["Which datasets are available?", "-p", "offline", "-d", str(data_dir)]) == 0
    assert "house-prices.csv" in capsys.readouterr().out


def test_cli_trace_shows_tool_loop(capsys, data_dir):
    main(["Train a model for each dataset", "-p", "offline", "-d", str(data_dir), "--trace"])
    out = capsys.readouterr().out
    for expected in ["Human", "list_csv_files", "get_dataset_summaries", "evaluate_regression_dataset", "AI answer"]:
        assert expected in out


def test_cli_interactive_keeps_history(monkeypatch, capsys, data_dir):
    answers = iter(["Which datasets are available?", "Train a model for each dataset", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["-p", "offline", "-d", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert out.count("DataWizard: ") == 2 and "see ya later" in out
