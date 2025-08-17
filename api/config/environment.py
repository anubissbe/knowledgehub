
# Enhanced environment configuration loading
import os
from dotenv import load_dotenv

# Load environment-specific configuration
env_file = f".env.{os.getenv('ENVIRONMENT', 'development')}"
if os.path.exists(env_file):
    load_dotenv(env_file)
    print(f"✅ Loaded environment configuration: {env_file}")
else:
    load_dotenv()  # Fallback to default .env
    print("⚠️ Using default environment configuration")

# Validate critical environment variables
required_vars = [
    'DATABASE_URL', 'REDIS_URL', 'JWT_SECRET_KEY'
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

print("✅ Environment configuration validated")
