import pytest

from canegap import auth, secrets_util as su
from canegap.providers import CDSEProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET", "CANEGAP_AUTH", "CANEGAP_ADMIN_USER", "CANEGAP_ADMIN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)


def _fake(monkeypatch, mapping):
    monkeypatch.setattr(su, "_streamlit_secrets", lambda: mapping)


def test_flat_and_nested_secrets_are_found(monkeypatch):
    _fake(monkeypatch, {"CDSE_CLIENT_ID": "flat-id", "CDSE_CLIENT_SECRET": "flat-secret"})
    p = CDSEProvider()
    assert p.configured and p.client_id == "flat-id" and p.client_secret == "flat-secret"
    assert su.source_of("CDSE_CLIENT_ID") == "Streamlit secrets (CDSE_CLIENT_ID)"
    _fake(monkeypatch, {"cdse": {"client_id": "n-id", "client_secret": "n-secret"}})
    p = CDSEProvider()
    assert p.configured and p.client_id == "n-id" and "[cdse] client_id" in su.source_of("CDSE_CLIENT_ID")


def test_environment_beats_secrets_and_source_never_reveals_values(monkeypatch):
    _fake(monkeypatch, {"CDSE_CLIENT_ID": "from-secrets", "CDSE_CLIENT_SECRET": "s"})
    monkeypatch.setenv("CDSE_CLIENT_ID", "from-env")
    assert CDSEProvider().client_id == "from-env" and su.source_of("CDSE_CLIENT_ID") == "environment variable"
    assert "from-env" not in str(su.source_of("CDSE_CLIENT_ID"))


def test_template_placeholders_and_blanks_count_as_missing(monkeypatch):
    _fake(monkeypatch, {"CDSE_CLIENT_ID": "PASTE-your-client-id", "CDSE_CLIENT_SECRET": "  "})
    p = CDSEProvider()
    assert not p.configured and su.source_of("CDSE_CLIENT_ID") is None
    with pytest.raises(Exception, match="not set"):
        p._auth()


def test_missing_secrets_file_is_not_an_error(monkeypatch):
    class Boom(dict):
        def __contains__(self, k):
            raise FileNotFoundError("no secrets.toml")
    _fake(monkeypatch, Boom())
    assert not CDSEProvider().configured and su.get_secret("CDSE_CLIENT_ID", "dflt") == "dflt"


def test_app_settings_can_come_from_secrets(monkeypatch, conn):
    _fake(monkeypatch, {"CANEGAP_AUTH": "off"})
    assert not auth.enabled()
    _fake(monkeypatch, {"CANEGAP_ADMIN_USER": "chief", "CANEGAP_ADMIN_PASSWORD": "Quartz-pass-12345"})
    assert auth.seed_admin_from_env(conn) and auth.get_user(conn, "chief")["role"] == "admin"


def test_example_file_is_valid_toml_and_has_no_live_values():
    import tomllib
    from pathlib import Path
    d = tomllib.loads(Path(__file__).resolve().parents[1].joinpath(".streamlit/secrets.toml.example").read_text())
    assert d["CDSE_CLIENT_ID"].startswith("PASTE") and d["CDSE_CLIENT_SECRET"].startswith("PASTE")
    assert su._usable(d["CDSE_CLIENT_ID"]) is None


def test_app_run_page_accepts_credentials_from_secrets(tmp_path, monkeypatch):
    """With secrets present the Run Analysis page stops showing the 'credentials not found' error."""
    from pathlib import Path
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from canegap import db, samples
    from canegap.geojson_import import validate_geojson
    st.cache_resource.clear()
    monkeypatch.setenv("CANEGAP_DATA_DIR", str(tmp_path)); monkeypatch.setenv("CANEGAP_AUTH", "off")
    conn = db.connect(tmp_path / "canegap.sqlite")
    db.save_import(conn, validate_geojson(samples.farm_with_fields(True), outline_rows=[6]), "s.geojson")
    app = str(Path(__file__).resolve().parents[1] / "app.py")

    def page(secrets):
        at = AppTest.from_file(app, default_timeout=60)
        if secrets:
            at.secrets["CDSE_CLIENT_ID"], at.secrets["CDSE_CLIENT_SECRET"] = "id", "secret"
        at.run(); at.sidebar.radio[0].set_value("Run Analysis").run()
        at.radio[[r.label for r in at.radio].index("Satellite source")].set_value("Sentinel-2 L2A (Copernicus CDSE)").run()
        assert not at.exception, at.exception
        return at

    no = page(False)
    assert any("credentials not found" in e.value.lower() for e in no.error) and any("secrets" in e.value for e in no.error)
    yes = page(True)
    assert not any("credentials not found" in e.value.lower() for e in yes.error)
    assert any("Credentials found in: Streamlit secrets" in c.value for c in yes.caption)
