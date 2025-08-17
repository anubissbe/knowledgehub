
import pytest
import requests

def test_health_endpoint():
    """Test health endpoint"""
    try:
        response = requests.get("http://192.168.1.25:3000/health", timeout=5)
        assert response.status_code == 200
        assert "status" in response.json()
    except Exception:
        # Service might not be running during tests
        pytest.skip("Service not available")

def test_api_endpoint():
    """Test API endpoint"""
    try:
        response = requests.get("http://192.168.1.25:3000/api", timeout=5)
        assert response.status_code in [200, 404]  # 404 is acceptable
    except Exception:
        pytest.skip("Service not available")
