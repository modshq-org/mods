use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
    /// Toggle the favorite state for an output path.
    /// Returns `true` if the image is now favorited, `false` if it was unfavorited.
    pub fn toggle_favorite(&self, path: &str) -> Result<bool> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM favorites WHERE path = ?1",
                params![path],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if exists {
            self.conn
                .execute("DELETE FROM favorites WHERE path = ?1", params![path])
                .context("Failed to remove favorite")?;
            Ok(false)
        } else {
            self.conn
                .execute("INSERT INTO favorites (path) VALUES (?1)", params![path])
                .context("Failed to insert favorite")?;
            Ok(true)
        }
    }

    /// Explicitly set the favorite state for an output path.
    /// Returns `true` if the state changed.
    pub fn set_favorite(&self, path: &str, favorited: bool) -> Result<bool> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM favorites WHERE path = ?1",
                params![path],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if favorited && !exists {
            self.conn
                .execute("INSERT INTO favorites (path) VALUES (?1)", params![path])
                .context("Failed to insert favorite")?;
            Ok(true)
        } else if !favorited && exists {
            self.conn
                .execute("DELETE FROM favorites WHERE path = ?1", params![path])
                .context("Failed to remove favorite")?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check whether a specific path is favorited.
    pub fn is_favorite(&self, path: &str) -> Result<bool> {
        let exists: bool = self
            .conn
            .query_row(
                "SELECT 1 FROM favorites WHERE path = ?1",
                params![path],
                |_| Ok(true),
            )
            .unwrap_or(false);
        Ok(exists)
    }

    /// Return all favorited output paths as a set.
    pub fn get_favorite_paths(&self) -> Result<std::collections::HashSet<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM favorites")
            .context("Failed to prepare favorites query")?;
        let paths = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .context("Failed to query favorites")?
            .collect::<std::result::Result<std::collections::HashSet<_>, _>>()
            .context("Failed to collect favorites")?;
        Ok(paths)
    }
}
