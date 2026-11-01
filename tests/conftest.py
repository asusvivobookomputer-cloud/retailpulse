import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retailpulse import analysis, etl, generate_data  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate_data.generate(d, n_customers=700, seed=7, start_date="2023-01-01", end_date="2025-09-30")
    return d


@pytest.fixture(scope="session")
def db(tmp_path_factory, raw_dir):
    path = tmp_path_factory.mktemp("db") / "test.db"
    etl.run_etl(raw_dir, path)
    return path


@pytest.fixture(scope="session")
def lines(db):
    conn = analysis.connect(db)
    yield analysis.load_lines(conn)
    conn.close()


@pytest.fixture(scope="session")
def customers(db):
    conn = analysis.connect(db)
    yield analysis.load_customers(conn)
    conn.close()
