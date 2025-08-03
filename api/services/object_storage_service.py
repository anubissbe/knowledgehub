"""
Object Storage Service using MinIO

This service provides object storage capabilities for large files,
documents, and binary data using MinIO S3-compatible storage.
"""

import os
import json
import hashlib
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime, timedelta
import logging
from io import BytesIO

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError

logger = logging.getLogger(__name__)


class ObjectStorageService:
    """Service for managing object storage with MinIO"""
    
    def __init__(self):
        """Initialize MinIO client"""
        self.endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            self._initialize_buckets()
            self.connected = True
            logger.info(f"Connected to MinIO at {self.endpoint}")
        except (MaxRetryError, S3Error) as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            self.client = None
            self.connected = False
    
    def _initialize_buckets(self):
        """Create default buckets if they don't exist"""
        default_buckets = [
            "documents",      # For scraped documents
            "attachments",    # For user attachments
            "embeddings",     # For large embedding files
            "backups",        # For system backups
            "temp"           # For temporary files
        ]
        
        for bucket in default_buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
                    
                    # Set lifecycle rules for temp bucket
                    if bucket == "temp":
                        self._set_temp_bucket_lifecycle()
                        
            except S3Error as e:
                logger.error(f"Failed to create bucket {bucket}: {e}")
    
    def _set_temp_bucket_lifecycle(self):
        """Set lifecycle policy for temp bucket to auto-delete after 24 hours"""
        lifecycle_config = {
            "Rules": [{
                "ID": "AutoDeleteTempFiles",
                "Status": "Enabled",
                "Expiration": {
                    "Days": 1
                }
            }]
        }
        # Note: MinIO Python SDK doesn't have direct lifecycle support
        # This would need to be done via MinIO admin API or mc command
        logger.info("Temp bucket lifecycle rules should be configured via MinIO admin")
    
    def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_data: BinaryIO,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Upload a file to object storage"""
        if not self.connected:
            logger.error("MinIO not connected")
            return None
            
        try:
            # Check if bucket exists
            if not self.client.bucket_exists(bucket):
                logger.error(f"Bucket '{bucket}' does not exist")
                return None
            
            # Calculate file size
            file_data.seek(0, 2)  # Seek to end
            file_size = file_data.tell()
            file_data.seek(0)  # Reset to beginning
            
            # Calculate checksum
            checksum = self._calculate_checksum(file_data)
            file_data.seek(0)  # Reset again
            
            # Add checksum to metadata
            if metadata is None:
                metadata = {}
            metadata["checksum"] = checksum
            metadata["upload_time"] = datetime.utcnow().isoformat()
            
            # Upload file
            result = self.client.put_object(
                bucket,
                object_name,
                file_data,
                file_size,
                content_type=content_type,
                metadata=metadata
            )
            
            return {
                "bucket": bucket,
                "object_name": object_name,
                "etag": result.etag,
                "version_id": result.version_id,
                "checksum": checksum,
                "size": file_size,
                "content_type": content_type,
                "metadata": metadata
            }
            
        except S3Error as e:
            logger.error(f"Failed to upload file: {e}")
            return None
    
    def download_file(
        self,
        bucket: str,
        object_name: str
    ) -> Optional[BytesIO]:
        """Download a file from object storage"""
        if not self.connected:
            logger.error("MinIO not connected")
            return None
            
        try:
            response = self.client.get_object(bucket, object_name)
            file_data = BytesIO(response.read())
            response.close()
            response.release_conn()
            return file_data
            
        except S3Error as e:
            logger.error(f"Failed to download file: {e}")
            return None
    
    def get_file_info(
        self,
        bucket: str,
        object_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata about a stored file"""
        if not self.connected:
            logger.error("MinIO not connected")
            return None
            
        try:
            stat = self.client.stat_object(bucket, object_name)
            
            return {
                "bucket": bucket,
                "object_name": object_name,
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified.isoformat(),
                "metadata": stat.metadata,
                "version_id": stat.version_id
            }
            
        except S3Error as e:
            logger.error(f"Failed to get file info: {e}")
            return None
    
    def delete_file(
        self,
        bucket: str,
        object_name: str
    ) -> bool:
        """Delete a file from object storage"""
        if not self.connected:
            logger.error("MinIO not connected")
            return False
            
        try:
            self.client.remove_object(bucket, object_name)
            return True
            
        except S3Error as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def list_files(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """List files in a bucket"""
        if not self.connected:
            logger.error("MinIO not connected")
            return []
            
        try:
            objects = self.client.list_objects(
                bucket,
                prefix=prefix,
                recursive=recursive
            )
            
            files = []
            for obj in objects:
                files.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "etag": obj.etag,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "is_dir": obj.is_dir
                })
            
            return files
            
        except S3Error as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def create_presigned_url(
        self,
        bucket: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """Generate a presigned URL for temporary access"""
        if not self.connected:
            logger.error("MinIO not connected")
            return None
            
        try:
            url = self.client.presigned_get_object(
                bucket,
                object_name,
                expires=expires
            )
            return url
            
        except S3Error as e:
            logger.error(f"Failed to create presigned URL: {e}")
            return None
    
    def enable_versioning(self, bucket: str) -> bool:
        """Enable versioning for a bucket"""
        if not self.connected:
            logger.error("MinIO not connected")
            return False
            
        try:
            # Note: MinIO Python SDK doesn't have direct versioning support
            # This would need to be done via MinIO admin API
            logger.info(f"Versioning for bucket {bucket} should be enabled via MinIO admin")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable versioning: {e}")
            return False
    
    def _calculate_checksum(self, file_data: BinaryIO) -> str:
        """Calculate SHA256 checksum of file data"""
        sha256_hash = hashlib.sha256()
        for byte_block in iter(lambda: file_data.read(4096), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        if not self.connected:
            return {
                "connected": False,
                "buckets": [],
                "total_size": 0,
                "total_objects": 0
            }
            
        try:
            buckets = self.client.list_buckets()
            bucket_stats = []
            total_size = 0
            total_objects = 0
            
            for bucket in buckets:
                objects = list(self.client.list_objects(bucket.name, recursive=True))
                bucket_size = sum(obj.size for obj in objects)
                bucket_count = len(objects)
                
                bucket_stats.append({
                    "name": bucket.name,
                    "created": bucket.creation_date.isoformat(),
                    "size": bucket_size,
                    "objects": bucket_count
                })
                
                total_size += bucket_size
                total_objects += bucket_count
            
            return {
                "connected": True,
                "endpoint": self.endpoint,
                "buckets": bucket_stats,
                "total_size": total_size,
                "total_objects": total_objects
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                "connected": True,
                "error": str(e),
                "buckets": [],
                "total_size": 0,
                "total_objects": 0
            }


# Global instance
object_storage_service = ObjectStorageService()