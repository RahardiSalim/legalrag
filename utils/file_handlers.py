import tempfile
import zipfile
import logging
from pathlib import Path
from typing import List
from fastapi import UploadFile

from config.settings import settings

logger = logging.getLogger(__name__)


class FileHandler:
    """Handle file operations for document upload"""
    
    def __init__(self):
        self.supported_extensions = settings.document.supported_extensions
        self.temp_dir = settings.storage.temp_dir
        self.max_file_size = settings.document.max_document_size
    
    def extract_files_from_zip(self, zip_path: str, extract_to: str) -> List[str]:
        """Extract supported files from ZIP archive"""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if not file_info.is_dir():
                        file_extension = Path(file_info.filename).suffix.lower()
                        
                        if file_extension in self.supported_extensions:
                            # Check file size before extraction
                            if file_info.file_size > self.max_file_size:
                                logger.warning(f"Skipping large file in ZIP: {file_info.filename} ({file_info.file_size} bytes)")
                                continue
                            
                            try:
                                extracted_path = zip_ref.extract(file_info, extract_to)
                                extracted_files.append(extracted_path)
                                logger.info(f"Extracted from ZIP: {file_info.filename}")
                            except Exception as e:
                                logger.error(f"Failed to extract {file_info.filename}: {e}")
                        else:
                            logger.debug(f"Skipping unsupported file in ZIP: {file_info.filename}")
            
            logger.info(f"Extracted {len(extracted_files)} files from ZIP: {zip_path}")
            
        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {zip_path}")
        except Exception as e:
            logger.error(f"Error extracting ZIP file {zip_path}: {e}")
        
        return extracted_files

    def save_uploaded_file(self, file: UploadFile, temp_dir: str) -> str:
        """Save uploaded file to temporary directory"""
        if not file.filename:
            raise ValueError("File has no filename")
        
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name
        file_path = Path(temp_dir) / safe_filename
        
        try:
            with open(file_path, 'wb') as f:
                # Read file in chunks to handle large files
                chunk_size = 8192
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            logger.info(f"Saved uploaded file: {safe_filename} ({file_path.stat().st_size} bytes)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file {safe_filename}: {e}")
            # Clean up partial file if it exists
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            raise

    def validate_file(self, file: UploadFile) -> bool:
        """Validate uploaded file"""
        if not file.filename:
            logger.warning("File validation failed: No filename")
            return False
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        is_supported = (file_extension in self.supported_extensions or 
                       file_extension == '.zip')
        
        if not is_supported:
            logger.warning(f"File validation failed: Unsupported extension {file_extension} for {file.filename}")
            return False
        
        # Check file size if available
        if hasattr(file, 'size') and file.size is not None:
            if file.size > self.max_file_size:
                logger.warning(f"File validation failed: File too large {file.filename} ({file.size} bytes)")
                return False
            elif file.size == 0:
                logger.warning(f"File validation failed: Empty file {file.filename}")
                return False
        
        logger.debug(f"File validation passed: {file.filename}")
        return True

    def process_uploaded_files(self, files: List[UploadFile]) -> List[str]:
        """Process list of uploaded files"""
        # Create temporary directory with proper permissions
        temp_dir = tempfile.mkdtemp(
            dir=self.temp_dir,
            prefix="upload_"
        )
        
        file_paths = []
        processed_count = 0
        
        try:
            for file in files:
                try:
                    # Reset file pointer to beginning
                    file.file.seek(0)
                    
                    file_path = self.save_uploaded_file(file, temp_dir)
                    
                    if file.filename and file.filename.lower().endswith('.zip'):
                        # Extract files from ZIP
                        extracted_files = self.extract_files_from_zip(file_path, temp_dir)
                        file_paths.extend(extracted_files)
                        processed_count += len(extracted_files)
                        
                        # Remove the ZIP file after extraction
                        Path(file_path).unlink(missing_ok=True)
                    else:
                        file_paths.append(file_path)
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process file {file.filename}: {e}")
                    continue
            
            logger.info(f"Successfully processed {processed_count} files from {len(files)} uploads")
            
        except Exception as e:
            logger.error(f"Error processing uploaded files: {e}")
            # Clean up temp directory on error
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        
        return file_paths
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        cleaned_count = 0
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    path.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
    
    def get_file_info(self, file_path: str) -> dict:
        """Get information about a file"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": "File not found"}
            
            stat = path.stat()
            return {
                "filename": path.name,
                "size": stat.st_size,
                "extension": path.suffix.lower(),
                "is_supported": path.suffix.lower() in self.supported_extensions,
                "created": stat.st_ctime,
                "modified": stat.st_mtime
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"error": str(e)}


class FileValidator:
    """Additional file validation utilities"""
    
    @staticmethod
    def is_safe_filename(filename: str) -> bool:
        """Check if filename is safe (no path traversal)"""
        if not filename:
            return False
        
        # Check for path traversal attempts
        dangerous_patterns = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        return not any(pattern in filename for pattern in dangerous_patterns)
    
    @staticmethod
    def get_file_type_info(file_extension: str) -> dict:
        """Get information about file type"""
        file_types = {
            '.pdf': {'name': 'PDF Document', 'category': 'document'},
            '.txt': {'name': 'Text File', 'category': 'text'},
            '.docx': {'name': 'Word Document', 'category': 'document'},
            '.md': {'name': 'Markdown File', 'category': 'text'},
            '.zip': {'name': 'ZIP Archive', 'category': 'archive'}
        }
        
        return file_types.get(file_extension.lower(), {
            'name': 'Unknown', 
            'category': 'unknown'
        })