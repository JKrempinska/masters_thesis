from masters import main


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from masters!" in captured.out
