use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
    /// Create a new studio session.
    pub fn create_studio_session(&self, id: &str, intent: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO studio_sessions (id, intent, status) VALUES (?1, ?2, 'pending')",
                params![id, intent],
            )
            .context("Failed to create studio session")?;
        Ok(())
    }

    /// Update studio session status.
    pub fn update_studio_session_status(&self, id: &str, status: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        if status == "completed" || status == "failed" {
            self.conn
                .execute(
                    "UPDATE studio_sessions SET status = ?1, completed_at = ?2 WHERE id = ?3",
                    params![status, now, id],
                )
                .context("Failed to update studio session")?;
        } else {
            self.conn
                .execute(
                    "UPDATE studio_sessions SET status = ?1 WHERE id = ?2",
                    params![status, id],
                )
                .context("Failed to update studio session")?;
        }
        Ok(())
    }

    /// Get a studio session by ID.
    pub fn get_studio_session(&self, id: &str) -> Result<Option<StudioSessionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, intent, status, created_at, completed_at FROM studio_sessions WHERE id = ?1",
        ).context("Failed to prepare query")?;

        let mut rows = stmt
            .query_map(params![id], StudioSessionRecord::from_row)
            .context("Failed to query studio session")?;

        match rows.next() {
            Some(Ok(record)) => Ok(Some(record)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// List all studio sessions, newest first.
    pub fn list_studio_sessions(&self) -> Result<Vec<StudioSessionRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, intent, status, created_at, completed_at FROM studio_sessions ORDER BY created_at DESC",
        ).context("Failed to prepare query")?;

        let rows = stmt
            .query_map([], StudioSessionRecord::from_row)
            .context("Failed to query studio sessions")?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect studio sessions")
    }

    /// Delete a studio session and its events/images.
    pub fn delete_studio_session(&self, id: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM session_events WHERE session_id = ?1",
                params![id],
            )
            .context("Failed to delete session events")?;
        self.conn
            .execute(
                "DELETE FROM session_images WHERE session_id = ?1",
                params![id],
            )
            .context("Failed to delete session images")?;
        self.conn
            .execute("DELETE FROM studio_sessions WHERE id = ?1", params![id])
            .context("Failed to delete studio session")?;
        Ok(())
    }

    /// Insert a session event.
    pub fn insert_session_event(
        &self,
        session_id: &str,
        sequence: u32,
        event_json: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO session_events (session_id, sequence, event_json) VALUES (?1, ?2, ?3)",
                params![session_id, sequence, event_json],
            )
            .context("Failed to insert session event")?;
        Ok(())
    }

    /// Get all events for a session, ordered by sequence.
    pub fn get_session_events(&self, session_id: &str) -> Result<Vec<SessionEventRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT session_id, sequence, event_json, timestamp FROM session_events WHERE session_id = ?1 ORDER BY sequence",
        ).context("Failed to prepare query")?;

        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok(SessionEventRecord {
                    session_id: row.get(0)?,
                    sequence: row.get(1)?,
                    event_json: row.get(2)?,
                    timestamp: row.get(3)?,
                })
            })
            .context("Failed to query session events")?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect session events")
    }

    /// Insert a session image reference.
    pub fn insert_session_image(
        &self,
        session_id: &str,
        image_path: &str,
        role: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO session_images (session_id, image_path, role) VALUES (?1, ?2, ?3)",
                params![session_id, image_path, role],
            )
            .context("Failed to insert session image")?;
        Ok(())
    }

    /// Get all images for a session, optionally filtered by role.
    pub fn get_session_images(&self, session_id: &str, role: Option<&str>) -> Result<Vec<String>> {
        let (sql, p): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(r) = role {
            (
                "SELECT image_path FROM session_images WHERE session_id = ?1 AND role = ?2",
                vec![Box::new(session_id.to_string()), Box::new(r.to_string())],
            )
        } else {
            (
                "SELECT image_path FROM session_images WHERE session_id = ?1",
                vec![Box::new(session_id.to_string())],
            )
        };

        let mut stmt = self.conn.prepare(sql).context("Failed to prepare query")?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(p.iter()), |row| {
                row.get::<_, String>(0)
            })
            .context("Failed to query session images")?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect session images")
    }
}

#[derive(Debug)]
pub struct StudioSessionRecord {
    pub id: String,
    pub intent: String,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
}

impl StudioSessionRecord {
    fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            intent: row.get(1)?,
            status: row.get(2)?,
            created_at: row.get(3)?,
            completed_at: row.get(4)?,
        })
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct SessionEventRecord {
    pub session_id: String,
    pub sequence: u32,
    pub event_json: String,
    pub timestamp: String,
}
