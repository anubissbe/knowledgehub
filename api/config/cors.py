
from fastapi.middleware.cors import CORSMiddleware

CORS_CONFIG = {
    "allow_origins": ["http://192.168.1.25:3100", "http://localhost:3100"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

def setup_cors(app):
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        **CORS_CONFIG
    )
