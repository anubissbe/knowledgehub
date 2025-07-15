# KnowledgeHub Installation Fixes

## Issues Found and Fixed

### 1. ✅ Docker Compose Compatibility (FIXED)
**Problem**: Original install.sh used `docker-compose` which fails on systems with Docker Compose v2  
**Solution**: Created `install_fixed.sh` that auto-detects the correct command

### 2. ✅ Port Mismatch in AI Service (FIXED)
**Problem**: AI service Dockerfile used port 8000 but docker-compose expected 8002  
**Solution**: Updated Dockerfile to use port 8002

### 3. ✅ Duplicate Dependencies (FIXED)
**Problem**: pandas==2.1.4 was listed twice in requirements.txt  
**Solution**: Removed duplicate entry

### 4. ⚠️ Hardcoded Paths (NEEDS ATTENTION)
**Problem**: Multiple files have hardcoded paths to `/opt/projects/memory-system/`  
**Temporary Solution**: The fixed installer creates these directories and adds path variables to .env

**Files that need updating**:
- `api/services/project_context_manager.py` (line 27)
- `api/services/claude_session_manager.py` (line 25)
- `api/services/memory_sync.py` (line 27)
- `api/memory_system/*.py` (multiple files with hardcoded paths)

### 5. ⚠️ External Dependencies
**Problem**: Some services expect external systems (ProjectHub at 192.168.1.24)  
**Impact**: These features will fail gracefully but won't work without external services

## Quick Start with Fixes

1. **Use the fixed installer**:
   ```bash
   chmod +x install_fixed.sh
   ./install_fixed.sh
   ```

2. **Update .env file** with your specific configuration

3. **For production**, update these hardcoded references:
   ```python
   # Change from:
   self.memory_cli_path = "/opt/projects/memory-system/memory-cli"
   
   # To:
   self.memory_cli_path = os.getenv("MEMORY_CLI_PATH", 
                                   os.path.join(os.getenv("KNOWLEDGEHUB_ROOT", ""), 
                                               "data/memory-system/memory-cli"))
   ```

## Verification Commands

After installation, verify everything is working:

```bash
# Check all services are running
docker compose ps  # or docker-compose ps

# Test API health
curl http://localhost:3000/health

# Check logs for any errors
docker compose logs -f api

# Access the UI
open http://localhost:3100
```

## Known Limitations

1. **Memory System Integration**: Requires additional setup if you want full memory-system features
2. **External Service Dependencies**: ProjectHub, Vault integration won't work without those services
3. **LAN-specific IPs**: Some features expect 192.168.1.x network

## Next Steps for Repository Owner

1. Replace `install.sh` with `install_fixed.sh`
2. Update all hardcoded paths to use environment variables
3. Add a pre-flight check script to validate environment
4. Test on fresh Ubuntu 22.04, macOS, and Windows WSL2
5. Update README with clearer system requirements

## Emergency Troubleshooting

If installation fails:

1. **Check Docker**:
   ```bash
   docker --version
   docker compose version  # or docker-compose --version
   ```

2. **Check ports aren't in use**:
   ```bash
   netstat -tlnp | grep -E "3000|3100|5433|6381|8090|7474|9010"
   ```

3. **Clean restart**:
   ```bash
   docker compose down -v  # WARNING: This removes all data
   rm -rf data/
   ./install_fixed.sh
   ```

4. **Manual service start** (for debugging):
   ```bash
   docker compose up postgres redis -d
   # Wait 30 seconds
   docker compose up api -d
   # Check logs
   docker compose logs -f api
   ```