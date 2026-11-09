"""Users, roles, farm-level access and an audit log (stored in the application's SQLite database).

Roles
  viewer   read-only: explorer, results, dashboards, imagery, exports
  analyst  viewer + import boundaries, run analyses, review/edit gaps, surveys, interventions, ML training
  admin    analyst + user management, delete models/surveys, audit log

Security properties: salted scrypt password hashes, constant-time comparison, generic login errors, lock-out after
repeated failures, minimum password policy, protection against removing the last active admin, audit trail.
Sessions live in Streamlit's server-side session state (no passwords or tokens are stored in the browser); run behind
HTTPS in production. For enterprise SSO consider Streamlit's native OIDC (`st.login`) in front of / instead of this.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

ROLES = ("viewer", "analyst", "admin")
PERMS = {"view": {"viewer", "analyst", "admin"}, "write": {"analyst", "admin"}, "admin": {"admin"}}
MAX_FAILED = 5
LOCK_MINUTES = 5
MIN_PW_LEN = 10
_N, _R, _P = 2 ** 14, 8, 1


class AuthError(ValueError):
    pass


def enabled() -> bool:
    return os.environ.get("CANEGAP_AUTH", "on").strip().lower() not in ("off", "0", "false", "no")


def can(role: str | None, perm: str) -> bool:
    return role in PERMS.get(perm, set())


def _now(now=None) -> datetime:
    return now or datetime.now(timezone.utc)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(h).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(h))
    except Exception:  # noqa: BLE001
        return False


_DUMMY = hash_password("dummy-password-for-timing")


def check_password_policy(password: str, username: str = "") -> None:
    if len(password) < MIN_PW_LEN:
        raise AuthError(f"Password must be at least {MIN_PW_LEN} characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("Password must contain letters and digits.")
    if username and username.lower() in password.lower():
        raise AuthError("Password must not contain the username.")


def _check_username(u: str) -> str:
    u = (u or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{3,64}", u):
        raise AuthError("Username must be 3-64 characters: letters, digits and _ . @ -")
    return u


def audit(conn, username: str | None, action: str, detail: str = "", now=None) -> None:
    conn.execute("INSERT INTO audit(ts,username,action,detail) VALUES(?,?,?,?)",
                 (_now(now).isoformat(timespec="seconds"), username, action, detail[:2000]))
    conn.commit()


def has_users(conn) -> bool:
    return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def _active_admins(conn) -> int:
    return conn.execute("SELECT COUNT(*) c FROM users WHERE role='admin' AND active=1").fetchone()["c"]


def create_user(conn, username: str, password: str, role: str = "viewer", display_name: str | None = None,
                all_farms: bool = True, farms: list[int] | None = None, must_change_pw: bool = False,
                actor: str | None = None) -> int:
    username = _check_username(username)
    if role not in ROLES:
        raise AuthError(f"Role must be one of {ROLES}")
    check_password_policy(password, username)
    if conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
        raise AuthError("That username already exists.")
    uid = conn.execute(
        "INSERT INTO users(username,display_name,role,pw_hash,active,all_farms,must_change_pw,created_at) VALUES(?,?,?,?,1,?,?,?)",
        (username, display_name or username, role, hash_password(password), int(all_farms), int(must_change_pw),
         _now().isoformat(timespec="seconds"))).lastrowid
    for f in (farms or []):
        conn.execute("INSERT OR IGNORE INTO user_farms(user_id,farm_pk) VALUES(?,?)", (uid, int(f)))
    conn.commit()
    audit(conn, actor or username, "user_created", f"{username} role={role}")
    return uid


def seed_admin_from_env(conn) -> bool:
    """Create the first admin from CANEGAP_ADMIN_USER / CANEGAP_ADMIN_PASSWORD (headless deployments)."""
    u, p = os.environ.get("CANEGAP_ADMIN_USER"), os.environ.get("CANEGAP_ADMIN_PASSWORD")
    if u and p and not has_users(conn):
        create_user(conn, u, p, "admin", actor="bootstrap")
        return True
    return False


def authenticate(conn, username: str, password: str, now=None) -> dict:
    """Return the user row as dict or raise AuthError with a deliberately generic message."""
    now_dt = _now(now)
    row = conn.execute("SELECT * FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
    generic = AuthError("Invalid username or password.")
    if row is None:
        verify_password(password or "", _DUMMY)           # equalise timing
        audit(conn, username, "login_failed", "unknown user", now_dt)
        raise generic
    if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > now_dt:
        audit(conn, row["username"], "login_blocked", "account locked", now_dt)
        raise AuthError("Too many failed attempts. Try again later.")
    ok = verify_password(password or "", row["pw_hash"])
    if not ok or not row["active"]:
        fails = row["failed_attempts"] + 1
        lock = (now_dt + timedelta(minutes=LOCK_MINUTES)).isoformat() if fails >= MAX_FAILED else None
        conn.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?", (0 if lock else fails, lock, row["id"]))
        conn.commit()
        audit(conn, row["username"], "login_failed", "bad password" if not ok else "inactive account", now_dt)
        raise generic
    conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE id=?",
                 (now_dt.isoformat(timespec="seconds"), row["id"]))
    conn.commit()
    audit(conn, row["username"], "login", "", now_dt)
    return get_user(conn, row["username"])


def get_user(conn, username: str) -> dict | None:
    r = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if r is None:
        return None
    d = {k: r[k] for k in ("id", "username", "display_name", "role", "active", "all_farms", "must_change_pw", "last_login")}
    d["farms"] = [x["farm_pk"] for x in conn.execute("SELECT farm_pk FROM user_farms WHERE user_id=?", (r["id"],))]
    return d


def list_users(conn):
    import pandas as pd
    return pd.read_sql_query("SELECT id,username,display_name,role,active,all_farms,last_login,created_at,locked_until "
                             "FROM users ORDER BY username", conn)


def change_password(conn, username: str, old: str | None, new: str, actor: str | None = None, admin_reset: bool = False):
    r = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if r is None:
        raise AuthError("No such user.")
    if not admin_reset and not verify_password(old or "", r["pw_hash"]):
        raise AuthError("Current password is incorrect.")
    check_password_policy(new, r["username"])
    conn.execute("UPDATE users SET pw_hash=?, must_change_pw=?, failed_attempts=0, locked_until=NULL WHERE id=?",
                 (hash_password(new), 1 if admin_reset else 0, r["id"]))
    conn.commit()
    audit(conn, actor or username, "password_reset" if admin_reset else "password_changed", username)


def set_role(conn, username: str, role: str, actor: str | None = None):
    if role not in ROLES:
        raise AuthError(f"Role must be one of {ROLES}")
    r = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if r is None:
        raise AuthError("No such user.")
    if r["role"] == "admin" and role != "admin" and r["active"] and _active_admins(conn) <= 1:
        raise AuthError("Cannot demote the last active admin.")
    conn.execute("UPDATE users SET role=? WHERE id=?", (role, r["id"])); conn.commit()
    audit(conn, actor, "role_changed", f"{username} -> {role}")


def set_active(conn, username: str, active: bool, actor: str | None = None):
    r = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if r is None:
        raise AuthError("No such user.")
    if not active and r["role"] == "admin" and r["active"] and _active_admins(conn) <= 1:
        raise AuthError("Cannot deactivate the last active admin.")
    conn.execute("UPDATE users SET active=? WHERE id=?", (int(active), r["id"])); conn.commit()
    audit(conn, actor, "user_activated" if active else "user_deactivated", username)


def set_farm_access(conn, username: str, all_farms: bool, farms: list[int] | None = None, actor: str | None = None):
    r = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if r is None:
        raise AuthError("No such user.")
    conn.execute("UPDATE users SET all_farms=? WHERE id=?", (int(all_farms), r["id"]))
    conn.execute("DELETE FROM user_farms WHERE user_id=?", (r["id"],))
    for f in (farms or []):
        conn.execute("INSERT OR IGNORE INTO user_farms(user_id,farm_pk) VALUES(?,?)", (r["id"], int(f)))
    conn.commit()
    audit(conn, actor, "farm_access_changed", f"{username} all={all_farms} farms={farms}")


def allowed_farms(user: dict | None) -> set[int] | None:
    """None = unrestricted. Admins are always unrestricted."""
    if user is None or user["role"] == "admin" or user["all_farms"]:
        return None
    return set(user["farms"])


def audit_log(conn, limit: int = 500):
    import pandas as pd
    return pd.read_sql_query("SELECT ts,username,action,detail FROM audit ORDER BY id DESC LIMIT ?", conn, params=(limit,))
