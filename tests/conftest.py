import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from canegap import db


@pytest.fixture()
def conn(tmp_path):
    return db.connect(tmp_path / "t.sqlite")
