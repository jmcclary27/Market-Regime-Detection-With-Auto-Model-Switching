from src.ingestion.run_ingestion import main


def test_run_ingestion_delegates_to_poll_job(monkeypatch) -> None:
    received: list[list[str]] = []

    def fake_poll(argv: list[str]) -> None:
        received.append(argv)

    monkeypatch.setattr("src.ingestion.run_ingestion.poll_market_data_main", fake_poll)

    main()

    assert received == [[]]
