"""Storage configuration for accurate metrics reporting"""

# Actual storage configuration for the Synology NAS
STORAGE_CONFIG = {
    "data_volume": {
        "path": "/opt",
        "total_tb": 11.0,  # 11TB volume
        "description": "Main data storage volume"
    },
    "home_volume": {
        "path": "/home", 
        "total_tb": 4.0,   # 4TB volume
        "description": "Home directory storage"
    },
    "root_volume": {
        "path": "/",
        "total_gb": 219.0,  # 219GB root
        "description": "System root partition"
    },
    "actual_data_path": "/opt/projects/knowledgehub/data",
    "primary_storage_tb": 11.0  # Use this for dashboard display
}