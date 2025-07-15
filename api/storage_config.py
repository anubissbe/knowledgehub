"""Storage configuration for accurate metrics reporting"""

import os
from pathlib import Path

# Get base path from environment or use default
KNOWLEDGEHUB_BASE = Path(os.environ.get('KNOWLEDGEHUB_BASE', Path(__file__).parent.parent))
DATA_PATH = os.environ.get('KNOWLEDGEHUB_DATA', str(KNOWLEDGEHUB_BASE / 'data'))

# Actual storage configuration for the deployment
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
    "actual_data_path": DATA_PATH,
    "primary_storage_tb": 11.0  # Use this for dashboard display
}