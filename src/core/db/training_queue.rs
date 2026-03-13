use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
    pub fn list_training_queue(&self) -> Result<Vec<TrainingQueueItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, position, name, spec_json, status, created_at
             FROM training_queue
             WHERE status = 'pending'
             ORDER BY position ASC",
        )?;
        let items = stmt
            .query_map([], TrainingQueueItem::from_row)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(items)
    }

    pub fn add_to_training_queue(&self, name: &str, spec_json: &str) -> Result<i64> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("Failed to begin transaction for add_to_training_queue")?;
        let max_pos: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(position), 0) FROM training_queue WHERE status = 'pending'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        tx.execute(
            "INSERT INTO training_queue (position, name, spec_json, status) VALUES (?1, ?2, ?3, 'pending')",
            params![max_pos + 1, name, spec_json],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()
            .context("Failed to commit add_to_training_queue")?;
        Ok(id)
    }

    pub fn remove_from_training_queue(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM training_queue WHERE id = ?1", params![id])
            .context("Failed to remove from training queue")?;
        Ok(())
    }

    pub fn update_training_queue_position(&self, id: i64, new_position: i64) -> Result<()> {
        self.conn
            .execute(
                "UPDATE training_queue SET position = ?1 WHERE id = ?2",
                params![new_position, id],
            )
            .context("Failed to update queue position")?;
        Ok(())
    }

    pub fn pop_training_queue(&self) -> Result<Option<TrainingQueueItem>> {
        let tx = self
            .conn
            .unchecked_transaction()
            .context("Failed to begin transaction for pop_training_queue")?;

        let item: Option<TrainingQueueItem> = tx
            .query_row(
                "SELECT id, position, name, spec_json, status, created_at
                 FROM training_queue
                 WHERE status = 'pending'
                 ORDER BY position ASC
                 LIMIT 1",
                [],
                TrainingQueueItem::from_row,
            )
            .ok();

        if let Some(ref item) = item {
            tx.execute(
                "UPDATE training_queue SET status = 'started' WHERE id = ?1",
                params![item.id],
            )?;
        }

        tx.commit().context("Failed to commit pop_training_queue")?;
        Ok(item)
    }
}

#[derive(Debug)]
pub struct TrainingQueueItem {
    pub id: i64,
    pub position: i64,
    pub name: String,
    pub spec_json: String,
    pub status: String,
    pub created_at: String,
}

impl TrainingQueueItem {
    pub(crate) fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            position: row.get(1)?,
            name: row.get(2)?,
            spec_json: row.get(3)?,
            status: row.get(4)?,
            created_at: row.get(5)?,
        })
    }
}
