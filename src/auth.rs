//! Who may use the server, kept in a SQLite database (default $XDG_DATA_HOME/qwenny/qwenny.db): an optional password
//! for the web page (none by default: the page is open), API keys, and whether the API requires one (off by default).
//! The password is an Argon2 hash; API keys and session tokens are random 256-bit values stored as SHA-256 hashes
//! (their entropy makes a slow hash pointless and the lookup per request fast).
use anyhow::{Context, Result, ensure};
use argon2::Argon2;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// The session cookie of the web page
pub const COOKIE: &str = "qwenny_session";
/// How long a login lasts
pub const SESSION_SECS: i64 = 30 * 24 * 3600;
/// The prefix of every API key (recognizable in configs and logs)
const KEY_PREFIX: &str = "qwk_";

/// $XDG_DATA_HOME (else ~/.local/share)/qwenny/qwenny.db
pub fn default_path() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("qwenny/qwenny.db")
}

pub struct Auth {
    db: Mutex<Connection>,
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha(s: &str) -> String {
    hex(&Sha256::digest(s.as_bytes()))
}

/// `N` bytes from the kernel's random source
fn random<const N: usize>() -> [u8; N] {
    use std::io::Read;
    let mut b = [0u8; N];
    std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).expect("/dev/urandom");
    b
}

/// 32 random bytes as 64 hex digits
fn random_hex() -> String {
    hex(&random::<32>())
}

fn hash_password(pw: &str) -> Result<String> {
    ensure!(pw.chars().count() >= 8, "the password needs 8 characters at least");
    let salt = SaltString::encode_b64(&random::<16>()).map_err(|e| anyhow::anyhow!("a salt: {e}"))?;
    Ok(Argon2::default().hash_password(pw.as_bytes(), &salt).map_err(|e| anyhow::anyhow!("hashing the password: {e}"))?
        .to_string())
}

impl Auth {
    /// The database at `path` (made with its directory when missing)
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
        }
        let db = Connection::open(path).with_context(|| format!("database {}", path.display()))?;
        // the server and `qwenny password` may use it at the same time
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS password (id INTEGER PRIMARY KEY CHECK (id = 1), hash TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, expires INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
                 shown TEXT NOT NULL, created INTEGER NOT NULL, last_used INTEGER);
             CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(Auth { db: Mutex::new(db) })
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.db.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn hash(&self) -> Result<Option<String>> {
        Ok(self.db().query_row("SELECT hash FROM password WHERE id = 1", [], |r| r.get(0)).optional()?)
    }

    /// Whether `password` is the one set
    fn password_ok(&self, password: &str) -> Result<bool> {
        Ok(self.hash()?.is_some_and(|hash| {
            PasswordHash::new(&hash).is_ok_and(|h| Argon2::default().verify_password(password.as_bytes(), &h).is_ok())
        }))
    }

    /// Whether a password is set (else the web page is open)
    pub fn has_password(&self) -> Result<bool> {
        Ok(self.hash()?.is_some())
    }

    /// Set the password; every login ends
    pub fn set_password(&self, password: &str) -> Result<()> {
        let hash = hash_password(password)?;
        let db = self.db();
        db.execute("INSERT OR REPLACE INTO password (id, hash) VALUES (1, ?1)", params![hash])?;
        db.execute("DELETE FROM sessions", [])?;
        Ok(())
    }

    /// A new session token for the right password, else None
    pub fn login(&self, password: &str) -> Result<Option<String>> {
        if !self.password_ok(password)? {
            return Ok(None);
        }
        Ok(Some(self.new_session()?))
    }

    fn new_session(&self) -> Result<String> {
        let token = random_hex();
        let db = self.db();
        db.execute("DELETE FROM sessions WHERE expires < ?1", params![now()])?;
        db.execute("INSERT INTO sessions (token_hash, expires) VALUES (?1, ?2)", params![sha(&token), now() + SESSION_SECS])?;
        Ok(token)
    }

    /// The first password (only while none is set) and a session for it
    pub fn setup(&self, password: &str) -> Result<Option<String>> {
        if self.has_password()? {
            return Ok(None);
        }
        self.set_password(password)?;
        Ok(Some(self.new_session()?))
    }

    pub fn session(&self, token: &str) -> Result<bool> {
        Ok(self.db().query_row("SELECT COUNT(*) FROM sessions WHERE token_hash = ?1 AND expires >= ?2",
                               params![sha(token), now()], |r| r.get::<_, i64>(0))? > 0)
    }

    pub fn logout(&self, token: &str) -> Result<()> {
        self.db().execute("DELETE FROM sessions WHERE token_hash = ?1", params![sha(token)])?;
        Ok(())
    }

    /// Change the password, given the current one; false when that is wrong. Other logins end, `keep` stays.
    pub fn change_password(&self, old: &str, new: &str, keep: &str) -> Result<bool> {
        if !self.password_ok(old)? {
            return Ok(false);
        }
        let hash = hash_password(new)?;
        let db = self.db();
        db.execute("UPDATE password SET hash = ?1 WHERE id = 1", params![hash])?;
        db.execute("DELETE FROM sessions WHERE token_hash != ?1", params![sha(keep)])?;
        Ok(true)
    }

    /// Remove the password, given it (the page is open again); false when it is wrong
    pub fn remove_password(&self, old: &str) -> Result<bool> {
        if !self.password_ok(old)? {
            return Ok(false);
        }
        let db = self.db();
        db.execute("DELETE FROM password", [])?;
        db.execute("DELETE FROM sessions", [])?;
        Ok(true)
    }

    /// Whether the API requires a key (else it is open)
    pub fn require_key(&self) -> Result<bool> {
        let v: Option<String> = self.db().query_row("SELECT value FROM settings WHERE key = 'require_key'", [], |r| r.get(0))
            .optional()?;
        Ok(v.as_deref() == Some("1"))
    }

    pub fn set_require_key(&self, on: bool) -> Result<()> {
        self.db().execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('require_key', ?1)",
                          params![if on { "1" } else { "0" }])?;
        Ok(())
    }

    /// A new API key named `name`: its id and the key (shown once; only its hash is kept)
    pub fn new_key(&self, name: &str) -> Result<(i64, String)> {
        ensure!(!name.trim().is_empty(), "an API key needs a name");
        let key = format!("{KEY_PREFIX}{}", random_hex());
        let shown = format!("{}…{}", &key[..KEY_PREFIX.len() + 4], &key[key.len() - 4..]);
        let db = self.db();
        db.execute("INSERT INTO api_keys (name, key_hash, shown, created) VALUES (?1, ?2, ?3, ?4)",
                   params![name.trim(), sha(&key), shown, now()])?;
        Ok((db.last_insert_rowid(), key))
    }

    /// The keys: id, name, the key shortened, created and last used (seconds since 1970)
    pub fn keys(&self) -> Result<Vec<Json>> {
        let db = self.db();
        let mut q = db.prepare("SELECT id, name, shown, created, last_used FROM api_keys ORDER BY id")?;
        let rows = q.query_map([], |r| {
            Ok(json!({"id": r.get::<_, i64>(0)?, "name": r.get::<_, String>(1)?, "key": r.get::<_, String>(2)?,
                      "created": r.get::<_, i64>(3)?, "last_used": r.get::<_, Option<i64>>(4)?}))
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Remove a key; false when there is none of that id
    pub fn revoke(&self, id: i64) -> Result<bool> {
        Ok(self.db().execute("DELETE FROM api_keys WHERE id = ?1", params![id])? > 0)
    }

    /// Whether `key` is a key of the database (then its last use is now)
    pub fn check_key(&self, key: &str) -> Result<bool> {
        if !key.starts_with(KEY_PREFIX) {
            return Ok(false);
        }
        Ok(self.db().execute("UPDATE api_keys SET last_used = ?1 WHERE key_hash = ?2", params![now(), sha(key)])? > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_sessions_and_keys() {
        let dir = std::env::temp_dir().join(format!("qwenny-auth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = Auth::open(&dir.join("t.db")).unwrap();
        assert!(!a.has_password().unwrap() && !a.require_key().unwrap());
        assert!(a.setup("short").is_err());
        let t = a.setup("correct horse").unwrap().unwrap();
        assert!(a.setup("another one").unwrap().is_none(), "setup only while no password is set");
        assert!(a.session(&t).unwrap() && !a.session("nope").unwrap());
        assert!(a.login("wrong password").unwrap().is_none());
        let t2 = a.login("correct horse").unwrap().unwrap();
        assert!(!a.change_password("wrong", "another one", &t).unwrap());
        assert!(a.change_password("correct horse", "another one", &t).unwrap());
        assert!(a.session(&t).unwrap() && !a.session(&t2).unwrap());
        a.logout(&t).unwrap();
        assert!(!a.session(&t).unwrap());
        let (id, key) = a.new_key("laptop").unwrap();
        assert!(key.starts_with(KEY_PREFIX) && a.check_key(&key).unwrap() && !a.check_key("qwk_00").unwrap());
        assert!(a.keys().unwrap()[0]["last_used"].is_i64());
        assert!(a.revoke(id).unwrap() && !a.check_key(&key).unwrap());
        assert!(!a.remove_password("correct horse").unwrap() && a.remove_password("another one").unwrap());
        assert!(!a.has_password().unwrap());
        a.set_require_key(true).unwrap();
        // a second connection (`qwenny password` next to a server) sees the change
        assert!(Auth::open(&dir.join("t.db")).unwrap().require_key().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
