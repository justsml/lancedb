use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PathAbs {
    path: PathBuf,
}

impl PathAbs {
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
        })
    }

    pub fn as_path(&self) -> &Path {
        &self.path
    }
}
