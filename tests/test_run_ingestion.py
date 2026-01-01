import pytest
from src.ingestion.run_ingestion import main

def test_run_ingestion_calls_fetch_and_raises():
    with pytest.raises(NotImplementedError):
        main()