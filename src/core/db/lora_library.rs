use anyhow::{Context, Result};
use rusqlite::params;

use super::Database;

impl Database {
    /// Promote a LoRA into the library.
    pub fn insert_library_lora(&self, record: &LibraryLoraRecord) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO lora_library
                 (id, name, trigger_word, base_model, lora_path, thumbnail, step, training_run, config_json, tags, notes, size_bytes)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    record.id,
                    record.name,
                    record.trigger_word,
                    record.base_model,
                    record.lora_path,
                    record.thumbnail,
                    record.step.map(|s| s as i64),
                    record.training_run,
                    record.config_json,
                    record.tags,
                    record.notes,
                    record.size_bytes as i64,
                ],
            )
            .context("Failed to insert library LoRA")?;
        Ok(())
    }

    /// List all promoted LoRAs.
    pub fn list_library_loras(&self) -> Result<Vec<LibraryLoraRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, name, trigger_word, base_model, lora_path, thumbnail, step, training_run, config_json, tags, notes, size_bytes, created_at
                 FROM lora_library ORDER BY created_at DESC",
            )
            .context("Failed to prepare library query")?;
        let rows = stmt
            .query_map([], LibraryLoraRecord::from_row)
            .context("Failed to query library LoRAs")?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to collect library LoRAs")
    }

    /// Get a single library LoRA by ID.
    #[allow(dead_code)]
    pub fn get_library_lora(&self, id: &str) -> Result<Option<LibraryLoraRecord>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, name, trigger_word, base_model, lora_path, thumbnail, step, training_run, config_json, tags, notes, size_bytes, created_at
                 FROM lora_library WHERE id = ?1",
            )
            .context("Failed to prepare library query")?;
        let mut rows = stmt
            .query_map(params![id], LibraryLoraRecord::from_row)
            .context("Failed to query library LoRA")?;
        match rows.next() {
            Some(r) => Ok(Some(r.context("Failed to read library LoRA")?)),
            None => Ok(None),
        }
    }

    /// Update a library LoRA's name, tags, or notes.
    pub fn update_library_lora(
        &self,
        id: &str,
        name: &str,
        tags: Option<&str>,
        notes: Option<&str>,
    ) -> Result<()> {
        self.conn
            .execute(
                "UPDATE lora_library SET name = ?2, tags = ?3, notes = ?4 WHERE id = ?1",
                params![id, name, tags, notes],
            )
            .context("Failed to update library LoRA")?;
        Ok(())
    }

    /// Update the lora_path for a library LoRA (e.g. after copying to store).
    pub fn update_library_lora_path(&self, id: &str, new_path: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE lora_library SET lora_path = ?2 WHERE id = ?1",
                params![id, new_path],
            )
            .context("Failed to update library LoRA path")?;
        Ok(())
    }

    /// Delete a library LoRA by ID.
    pub fn delete_library_lora(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM lora_library WHERE id = ?1", params![id])
            .context("Failed to delete library LoRA")?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LibraryLoraRecord {
    pub id: String,
    pub name: String,
    pub trigger_word: Option<String>,
    pub base_model: Option<String>,
    pub lora_path: String,
    pub thumbnail: Option<String>,
    pub step: Option<u64>,
    pub training_run: Option<String>,
    pub config_json: Option<String>,
    pub tags: Option<String>,
    pub notes: Option<String>,
    pub size_bytes: u64,
    pub created_at: String,
}

impl LibraryLoraRecord {
    pub(crate) fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            name: row.get(1)?,
            trigger_word: row.get(2)?,
            base_model: row.get(3)?,
            lora_path: row.get(4)?,
            thumbnail: row.get(5)?,
            step: row.get::<_, Option<i64>>(6)?.map(|v| v as u64),
            training_run: row.get(7)?,
            config_json: row.get(8)?,
            tags: row.get(9)?,
            notes: row.get(10)?,
            size_bytes: row.get::<_, i64>(11)? as u64,
            created_at: row.get(12)?,
        })
    }
}
