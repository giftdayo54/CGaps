"""Read configuration/credentials from environment variables or Streamlit secrets (never from code).

Lookup order for a name such as CDSE_CLIENT_ID:
  1. environment variable CDSE_CLIENT_ID
  2. Streamlit secrets, flat key       CDSE_CLIENT_ID = "..."
  3. Streamlit secrets, nested table   [cdse] client_id = "..."   (for the CDSE_* names)
Values that are empty or still look like template placeholders are treated as missing. Secret values are never logged
or displayed; `source_of` only reports WHERE a value was found.
"""
from __future__ import annotations

import os

NESTED = {"CDSE_CLIENT_ID": ("cdse", "client_id"), "CDSE_CLIENT_SECRET": ("cdse", "client_secret"),
          "CANEGAP_ADMIN_USER": ("canegap", "admin_user"), "CANEGAP_ADMIN_PASSWORD": ("canegap", "admin_password"),
          "CANEGAP_AUTH": ("canegap", "auth"), "CANEGAP_DATA_DIR": ("canegap", "data_dir")}


def _streamlit_secrets():
    """The st.secrets mapping, or {} when no secrets file/section exists (or Streamlit is not running)."""
    try:
        import streamlit as st
        return st.secrets
    except Exception:  # noqa: BLE001
        return {}


def _usable(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.upper().startswith(("PASTE", "YOUR-", "YOUR_", "CHANGE-ME", "CHANGEME")):
        return None
    return s


def _from_secrets(name: str):
    try:
        sec = _streamlit_secrets()
        if name in sec:
            return _usable(sec[name]), f"Streamlit secrets ({name})"
        if name in NESTED:
            table, key = NESTED[name]
            if table in sec and key in sec[table]:
                return _usable(sec[table][key]), f"Streamlit secrets ([{table}] {key})"
    except Exception:  # noqa: BLE001  (e.g. StreamlitSecretNotFoundError when no secrets.toml exists)
        pass
    return None, None


def get_secret(name: str, default: str | None = None) -> str | None:
    v = _usable(os.environ.get(name))
    if v is not None:
        return v
    v, _ = _from_secrets(name)
    return v if v is not None else default


def source_of(name: str) -> str | None:
    """Where a value was found ('environment variable', 'Streamlit secrets (...)') or None. Never returns the value."""
    if _usable(os.environ.get(name)) is not None:
        return "environment variable"
    v, src = _from_secrets(name)
    return src if v is not None else None
