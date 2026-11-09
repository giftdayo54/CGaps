from datetime import datetime, timedelta, timezone

import pytest

from canegap import auth, db

T0 = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def test_hash_roundtrip_unique_salts():
    h1, h2 = auth.hash_password("S3cure-pass-1"), auth.hash_password("S3cure-pass-1")
    assert h1 != h2 and auth.verify_password("S3cure-pass-1", h1) and not auth.verify_password("wrong", h1)
    assert not auth.verify_password("x", "garbage")


def test_password_policy_and_username_rules(conn):
    for bad in ("short1", "alllettersnodigit", "1234567890123"):
        with pytest.raises(auth.AuthError):
            auth.create_user(conn, "alice", bad)
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "alice", "alice-pass-12345")          # contains username
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "a b", "Valid-pass-123")
    auth.create_user(conn, "Alice", "Valid-pass-123", "analyst")
    with pytest.raises(auth.AuthError):
        auth.create_user(conn, "alice", "Other-pass-123")            # case-insensitive duplicate


def test_login_success_failure_generic_message_and_audit(conn):
    auth.create_user(conn, "bob", "Valid-pass-123", "viewer")
    assert auth.authenticate(conn, "BOB", "Valid-pass-123", T0)["role"] == "viewer"
    msgs = set()
    for u, p in (("bob", "wrong-pass-999"), ("ghost", "whatever-123")):
        with pytest.raises(auth.AuthError) as e:
            auth.authenticate(conn, u, p, T0)
        msgs.add(str(e.value))
    assert msgs == {"Invalid username or password."}                 # no user enumeration
    log = auth.audit_log(conn)
    assert {"login", "login_failed", "user_created"} <= set(log.action)


def test_lockout_and_unlock_after_window(conn):
    auth.create_user(conn, "carol", "Valid-pass-123", "analyst")
    for _ in range(auth.MAX_FAILED):
        with pytest.raises(auth.AuthError):
            auth.authenticate(conn, "carol", "bad-pass-1234", T0)
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.authenticate(conn, "carol", "Valid-pass-123", T0 + timedelta(minutes=1))   # right pw still blocked
    assert auth.authenticate(conn, "carol", "Valid-pass-123", T0 + timedelta(minutes=auth.LOCK_MINUTES + 1))["username"] == "carol"


def test_inactive_user_cannot_login_and_last_admin_protected(conn):
    auth.create_user(conn, "root", "Valid-pass-123", "admin")
    auth.create_user(conn, "dan", "Valid-pass-123", "analyst")
    auth.set_active(conn, "dan", False, "root")
    with pytest.raises(auth.AuthError):
        auth.authenticate(conn, "dan", "Valid-pass-123", T0)
    with pytest.raises(auth.AuthError, match="last active admin"):
        auth.set_role(conn, "root", "viewer", "root")
    with pytest.raises(auth.AuthError, match="last active admin"):
        auth.set_active(conn, "root", False, "root")
    auth.create_user(conn, "root2", "Valid-pass-123", "admin")
    auth.set_role(conn, "root", "viewer", "root2")                   # allowed now


def test_permissions_matrix():
    assert auth.can("viewer", "view") and not auth.can("viewer", "write") and not auth.can("viewer", "admin")
    assert auth.can("analyst", "write") and not auth.can("analyst", "admin")
    assert auth.can("admin", "admin") and not auth.can(None, "view")


def test_change_and_reset_password(conn):
    auth.create_user(conn, "erin", "Valid-pass-123", "viewer")
    with pytest.raises(auth.AuthError):
        auth.change_password(conn, "erin", "wrong-old-111", "New-pass-12345")
    auth.change_password(conn, "erin", "Valid-pass-123", "New-pass-12345")
    assert auth.authenticate(conn, "erin", "New-pass-12345", T0)
    auth.change_password(conn, "erin", None, "Reset-pass-12345", actor="root", admin_reset=True)
    u = auth.authenticate(conn, "erin", "Reset-pass-12345", T0)
    assert u["must_change_pw"] == 1


def test_farm_access_scoping(conn):
    auth.create_user(conn, "root", "Valid-pass-123", "admin")
    auth.create_user(conn, "fay", "Valid-pass-123", "analyst", all_farms=False, farms=[2, 3])
    fay = auth.get_user(conn, "fay")
    assert auth.allowed_farms(fay) == {2, 3} and auth.allowed_farms(auth.get_user(conn, "root")) is None
    auth.set_farm_access(conn, "fay", True, actor="root")
    assert auth.allowed_farms(auth.get_user(conn, "fay")) is None


def test_env_seed_and_toggle(conn, monkeypatch):
    monkeypatch.setenv("CANEGAP_ADMIN_USER", "boss"); monkeypatch.setenv("CANEGAP_ADMIN_PASSWORD", "Quartz-pass-12345")
    assert auth.seed_admin_from_env(conn) and not auth.seed_admin_from_env(conn)      # only on first run
    assert auth.get_user(conn, "boss")["role"] == "admin"
    monkeypatch.setenv("CANEGAP_AUTH", "off"); assert not auth.enabled()
    monkeypatch.delenv("CANEGAP_AUTH"); assert auth.enabled()
