//! Execution checkpointing for chunked model proving.
//!
//! Saves and loads intermediate activations between proving chunks,
//! enabling large models to be proven block-by-block with bounded memory.

use std::path::Path;

use stwo::core::fields::m31::M31;

use crate::components::matmul::M31Matrix;

/// A snapshot of the activation tensor at a specific point in the graph.
///
/// Persisted between proving chunks so the next chunk can resume from
/// the correct intermediate state.
#[derive(Debug, Clone)]
pub struct ExecutionCheckpoint {
    /// The node ID after which this checkpoint was taken.
    pub node_id: usize,
    /// Number of rows in the activation matrix.
    pub rows: usize,
    /// Number of columns in the activation matrix.
    pub cols: usize,
    /// Raw M31 values as u32 (for serialization).
    pub data: Vec<u32>,
}

impl ExecutionCheckpoint {
    /// Create a checkpoint from an M31Matrix.
    pub fn from_matrix(node_id: usize, matrix: &M31Matrix) -> Self {
        Self {
            node_id,
            rows: matrix.rows,
            cols: matrix.cols,
            data: matrix.data.iter().map(|v| v.0).collect(),
        }
    }

    /// Convert this checkpoint back into an M31Matrix.
    pub fn to_matrix(&self) -> M31Matrix {
        let mut matrix = M31Matrix::new(self.rows, self.cols);
        for (i, &val) in self.data.iter().enumerate() {
            matrix.data[i] = M31::from(val);
        }
        matrix
    }

    /// Save checkpoint to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let data_str: Vec<String> = self.data.iter().map(|v| v.to_string()).collect();
        let json = format!(
            r#"{{"node_id":{},"rows":{},"cols":{},"data":[{}]}}"#,
            self.node_id,
            self.rows,
            self.cols,
            data_str.join(","),
        );
        std::fs::write(path, json)
    }

    /// Load checkpoint from a JSON file.
    ///
    /// Uses a simple parser to avoid requiring serde_json as a non-optional dependency.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(path)?;

        // Simple extraction — the format is fixed and controlled by save()
        let node_id = extract_usize(&contents, "node_id")?;
        let rows = extract_usize(&contents, "rows")?;
        let cols = extract_usize(&contents, "cols")?;

        // Extract data array
        let data_start = contents.find("\"data\":[").ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing data field")
        })? + 8;
        let data_end = contents[data_start..].find(']').ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "unclosed data array")
        })? + data_start;

        let data_str = &contents[data_start..data_end];
        let data: Vec<u32> = if data_str.is_empty() {
            Vec::new()
        } else {
            data_str
                .split(',')
                .map(|s| {
                    s.trim().parse::<u32>().map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("bad u32: {e}"),
                        )
                    })
                })
                .collect::<std::io::Result<Vec<u32>>>()?
        };

        Ok(Self {
            node_id,
            rows,
            cols,
            data,
        })
    }
}

/// Extract a usize value from a JSON string by key name.
fn extract_usize(json: &str, key: &str) -> std::io::Result<usize> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("missing key '{key}'"),
        )
    })? + pattern.len();

    let end = json[start..]
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(json.len() - start)
        + start;

    json[start..end].trim().parse::<usize>().map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad usize: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut matrix = M31Matrix::new(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                matrix.set(i, j, M31::from((i * 3 + j + 1) as u32));
            }
        }

        let checkpoint = ExecutionCheckpoint::from_matrix(5, &matrix);
        assert_eq!(checkpoint.node_id, 5);
        assert_eq!(checkpoint.rows, 2);
        assert_eq!(checkpoint.cols, 3);
        assert_eq!(checkpoint.data.len(), 6);

        let recovered = checkpoint.to_matrix();
        assert_eq!(recovered.rows, 2);
        assert_eq!(recovered.cols, 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(recovered.get(i, j), matrix.get(i, j));
            }
        }
    }

    #[test]
    fn test_checkpoint_save_load() {
        let mut matrix = M31Matrix::new(1, 4);
        for j in 0..4 {
            matrix.set(0, j, M31::from((j * 10 + 42) as u32));
        }

        let checkpoint = ExecutionCheckpoint::from_matrix(3, &matrix);

        let dir = std::env::temp_dir().join("stwo_ml_checkpoint_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_ckpt.json");

        checkpoint.save(&path).expect("save should succeed");
        let loaded = ExecutionCheckpoint::load(&path).expect("load should succeed");

        assert_eq!(loaded.node_id, 3);
        assert_eq!(loaded.rows, 1);
        assert_eq!(loaded.cols, 4);
        assert_eq!(loaded.data, checkpoint.data);

        let recovered = loaded.to_matrix();
        for j in 0..4 {
            assert_eq!(recovered.get(0, j), matrix.get(0, j));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_empty_matrix() {
        let matrix = M31Matrix::new(0, 0);
        let checkpoint = ExecutionCheckpoint::from_matrix(0, &matrix);
        assert_eq!(checkpoint.data.len(), 0);
        let recovered = checkpoint.to_matrix();
        assert_eq!(recovered.rows, 0);
        assert_eq!(recovered.cols, 0);
    }
}
