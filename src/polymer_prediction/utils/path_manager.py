"""Cross-platform path management utilities."""

import os
import sys
from pathlib import Path
from typing import Union, Optional, List
from polymer_prediction.utils.logging import get_logger

logger = get_logger(__name__)


class PathManager:
    """Cross-platform path management utility."""
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """Initialize path manager.
        
        Args:
            base_path: Base path for all operations. Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.platform = sys.platform
        
        logger.info(f"PathManager initialized with base path: {self.base_path}")
        logger.info(f"Platform: {self.platform}")
    
    def resolve_path(self, path: Union[str, Path], create_parent: bool = False) -> Path:
        """Resolve a path relative to the base path.
        
        Args:
            path: Path to resolve
            create_parent: Whether to create parent directories
            
        Returns:
            Resolved Path object
        """
        if isinstance(path, str):
            path = Path(path)
        
        # If path is absolute, use as-is, otherwise make relative to base_path
        if path.is_absolute():
            resolved_path = path
        else:
            resolved_path = self.base_path / path
        
        # Resolve to absolute path
        resolved_path = resolved_path.resolve()
        
        if create_parent:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        return resolved_path
    
    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            Resolved directory path
        """
        dir_path = self.resolve_path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Directory ensured: {dir_path}")
        return dir_path
    
    def get_data_path(self, filename: str) -> Path:
        """Get path to a data file.
        
        Args:
            filename: Name of the data file
            
        Returns:
            Path to the data file
        """
        return self.resolve_path(Path("info") / filename)
    
    def get_model_path(self, filename: str) -> Path:
        """Get path to a model file.
        
        Args:
            filename: Name of the model file
            
        Returns:
            Path to the model file
        """
        model_dir = self.ensure_directory("models")
        return model_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get path to an output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Path to the output file
        """
        output_dir = self.ensure_directory("outputs")
        return output_dir / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get path to a log file.
        
        Args:
            filename: Name of the log file
            
        Returns:
            Path to the log file
        """
        log_dir = self.ensure_directory("logs")
        return log_dir / filename
    
    def get_cache_path(self, filename: str) -> Path:
        """Get path to a cache file.
        
        Args:
            filename: Name of the cache file
            
        Returns:
            Path to the cache file
        """
        cache_dir = self.ensure_directory("cache")
        return cache_dir / filename
    
    def get_checkpoint_path(self, filename: str) -> Path:
        """Get path to a checkpoint file.
        
        Args:
            filename: Name of the checkpoint file
            
        Returns:
            Path to the checkpoint file
        """
        checkpoint_dir = self.ensure_directory("checkpoints")
        return checkpoint_dir / filename
    
    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> List[Path]:
        """List files in a directory matching a pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            List of matching file paths
        """
        dir_path = self.resolve_path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []
        
        if not dir_path.is_dir():
            logger.warning(f"Path is not a directory: {dir_path}")
            return []
        
        files = list(dir_path.glob(pattern))
        files = [f for f in files if f.is_file()]
        
        logger.debug(f"Found {len(files)} files matching '{pattern}' in {dir_path}")
        return files
    
    def file_exists(self, path: Union[str, Path]) -> bool:
        """Check if a file exists.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        file_path = self.resolve_path(path)
        exists = file_path.exists() and file_path.is_file()
        
        if not exists:
            logger.debug(f"File does not exist: {file_path}")
        
        return exists
    
    def directory_exists(self, path: Union[str, Path]) -> bool:
        """Check if a directory exists.
        
        Args:
            path: Directory path to check
            
        Returns:
            True if directory exists, False otherwise
        """
        dir_path = self.resolve_path(path)
        exists = dir_path.exists() and dir_path.is_dir()
        
        if not exists:
            logger.debug(f"Directory does not exist: {dir_path}")
        
        return exists
    
    def get_file_size(self, path: Union[str, Path]) -> int:
        """Get file size in bytes.
        
        Args:
            path: File path
            
        Returns:
            File size in bytes, or 0 if file doesn't exist
        """
        file_path = self.resolve_path(path)
        
        if not self.file_exists(file_path):
            return 0
        
        return file_path.stat().st_size
    
    def get_relative_path(self, path: Union[str, Path]) -> Path:
        """Get path relative to base path.
        
        Args:
            path: Path to make relative
            
        Returns:
            Relative path
        """
        abs_path = self.resolve_path(path)
        
        try:
            return abs_path.relative_to(self.base_path)
        except ValueError:
            # Path is not relative to base_path, return as-is
            return abs_path
    
    def cleanup_directory(self, directory: Union[str, Path], pattern: str = "*", 
                         max_files: Optional[int] = None) -> int:
        """Clean up files in a directory.
        
        Args:
            directory: Directory to clean
            pattern: File pattern to match for deletion
            max_files: Maximum number of files to keep (newest files are kept)
            
        Returns:
            Number of files deleted
        """
        dir_path = self.resolve_path(directory)
        
        if not self.directory_exists(dir_path):
            logger.warning(f"Directory does not exist for cleanup: {dir_path}")
            return 0
        
        files = self.list_files(dir_path, pattern)
        
        if not files:
            logger.debug(f"No files found for cleanup in {dir_path}")
            return 0
        
        deleted_count = 0
        
        if max_files is not None and len(files) > max_files:
            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            files_to_delete = files[max_files:]
            
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")
        
        logger.info(f"Cleanup completed: {deleted_count} files deleted from {dir_path}")
        return deleted_count
    
    def get_platform_specific_path(self, unix_path: str, windows_path: str) -> Path:
        """Get platform-specific path.
        
        Args:
            unix_path: Path for Unix-like systems
            windows_path: Path for Windows systems
            
        Returns:
            Platform-appropriate path
        """
        if self.platform.startswith('win'):
            return self.resolve_path(windows_path)
        else:
            return self.resolve_path(unix_path)
    
    def get_temp_path(self, filename: str) -> Path:
        """Get path to a temporary file.
        
        Args:
            filename: Name of the temporary file
            
        Returns:
            Path to the temporary file
        """
        import tempfile
        
        temp_dir = Path(tempfile.gettempdir()) / "polymer_prediction"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        return temp_dir / filename


# Create a default path manager instance
default_path_manager = PathManager()

# Convenience functions using the default path manager
def resolve_path(path: Union[str, Path], create_parent: bool = False) -> Path:
    """Resolve a path using the default path manager."""
    return default_path_manager.resolve_path(path, create_parent)

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists using the default path manager."""
    return default_path_manager.ensure_directory(path)

def get_data_path(filename: str) -> Path:
    """Get data file path using the default path manager."""
    return default_path_manager.get_data_path(filename)

def get_model_path(filename: str) -> Path:
    """Get model file path using the default path manager."""
    return default_path_manager.get_model_path(filename)

def get_output_path(filename: str) -> Path:
    """Get output file path using the default path manager."""
    return default_path_manager.get_output_path(filename)

def get_log_path(filename: str) -> Path:
    """Get log file path using the default path manager."""
    return default_path_manager.get_log_path(filename)