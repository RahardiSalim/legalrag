import tempfile
import zipfile
from pathlib import Path
from typing import List
from fastapi import UploadFile

from config import Config

class FileHandler:
    def __init__(self, config: Config):
        self.config = config
    
    def extract_files_from_zip(self, zip_path: str, extract_to: str) -> List[str]:
        extracted_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():
                    file_extension = Path(file_info.filename).suffix.lower()
                    if file_extension in self.config.SUPPORTED_EXTENSIONS:
                        extracted_path = zip_ref.extract(file_info, extract_to)
                        extracted_files.append(extracted_path)
        
        return extracted_files

    def save_uploaded_file(self, file: UploadFile, temp_dir: str) -> str:
        file_path = Path(temp_dir) / file.filename
        
        with open(file_path, 'wb') as f:
            content = file.file.read()
            f.write(content)
        
        return str(file_path)

    def validate_file(self, file: UploadFile) -> bool:
        if not file.filename:
            return False
        
        file_extension = Path(file.filename).suffix.lower()
        return file_extension in self.config.SUPPORTED_EXTENSIONS or file_extension == '.zip'

    def process_uploaded_files(self, files: List[UploadFile]) -> List[str]:
        temp_dir = tempfile.mkdtemp(dir=self.config.TEMP_DIR)
        file_paths = []
        
        for file in files:
            file_path = self.save_uploaded_file(file, temp_dir)
            
            if file.filename.endswith('.zip'):
                extracted_files = self.extract_files_from_zip(file_path, temp_dir)
                file_paths.extend(extracted_files)
            else:
                file_paths.append(file_path)
        
        return file_paths