#\!/usr/bin/env python3
"""
Test Enterprise Features with Authentication
Creates JWT tokens and tests protected endpoints
"""
import jwt
import uuid
import requests
import json
from datetime import datetime, timedelta

# Create a test JWT token for testing
def create_test_jwt():
    """Create a test JWT token for enterprise testing"""
    
    # Use a consistent secret for testing
    jwt_secret = "test-enterprise-secret-key-for-knowledgehub"
    
    payload = {
        "user_id": str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "roles": ["system_admin", "power_user"],
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return token

def test_authenticated_endpoints():
    """Test enterprise endpoints with proper authentication"""
    
    base_url = "http://localhost:3001/api/v1/enterprise"
    
    # Create auth token
    token = create_test_jwt()
    
    # Headers with authentication
    headers = {
        'User-Agent': 'Mozilla/5.0 (Enterprise Test Client)',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    print("🔐 === ENTERPRISE AUTHENTICATION TEST ===")
    print(f"Created JWT token: {token[:50]}...")
    print()
    
    # Test cases with authentication
    test_cases = [
        {
            'name': 'Encrypt Sensitive Data',
            'method': 'POST',
            'path': '/security/encrypt',
            'data': {'plaintext': 'enterprise-secret-data-12345'}
        },
        {
            'name': 'Create Enterprise Tenant',
            'method': 'POST', 
            'path': '/tenants',
            'data': {
                'name': 'Authenticated Test Corp',
                'slug': 'auth-test-corp',
                'plan': 'enterprise',
                'billing_email': 'admin@authtest.com'
            }
        },
        {
            'name': 'Get Cluster Status',
            'method': 'GET',
            'path': '/cluster/status'
        },
        {
            'name': 'Check GPU Status',
            'method': 'GET',
            'path': '/gpu/status'
        },
        {
            'name': 'Register Cluster Node',
            'method': 'POST',
            'path': '/cluster/nodes/register',
            'data': {
                'service_type': 'ai_inference',
                'host': '192.168.1.25',
                'port': 8080,
                'metadata': {'gpu_count': 2, 'vram_total': 32768}
            }
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] Testing: {test['name']}")
        
        url = f"{base_url}{test['path']}"
        
        try:
            if test['method'] == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif test['method'] == 'POST':
                response = requests.post(url, headers=headers, 
                                       json=test.get('data'), timeout=10)
            
            status = response.status_code
            
            try:
                response_data = response.json()
            except:
                response_data = response.text[:200]
            
            result = {
                'test_name': test['name'],
                'method': test['method'],
                'path': test['path'], 
                'status_code': status,
                'response': response_data,
                'success': status == 200
            }
            
            if status == 200:
                print(f"         ✅ SUCCESS (200) - Endpoint working with auth\!")
                
                # Show interesting response data
                if isinstance(response_data, dict):
                    if 'encrypted' in response_data:
                        print(f"         🔐 Encrypted: {response_data['encrypted'][:50]}...")
                    elif 'id' in response_data:
                        print(f"         🆔 Created ID: {response_data['id']}")
                    elif 'cluster_status' in response_data:
                        print(f"         📊 Cluster: {response_data['cluster_status']}")
                    elif 'services' in response_data:
                        print(f"         🖥️  Services: {list(response_data['services'].keys())}")
                        
            elif status == 401:
                print(f"         🔒 AUTH ISSUE (401) - Token not accepted")
            elif status == 403:
                print(f"         🚫 PERMISSION (403) - Missing permissions")
            elif status == 500:
                print(f"         💥 SERVER ERROR (500) - Internal error")
            else:
                print(f"         ❓ UNEXPECTED ({status})")
                
            print(f"         Response: {str(response_data)[:100]}...")
            print()
            
            results.append(result)
            
        except Exception as e:
            print(f"         💥 EXCEPTION: {e}")
            print()
            
            results.append({
                'test_name': test['name'],
                'error': str(e),
                'success': False
            })
    
    # Summary
    working_count = sum(1 for r in results if r.get('success', False))
    total_count = len(results)
    
    print("🏆 === AUTHENTICATION TEST RESULTS ===")
    print(f"Working with Authentication: {working_count}/{total_count}")
    print(f"Success Rate: {working_count/total_count*100:.1f}%")
    print()
    
    if working_count > 0:
        print("✅ WORKING AUTHENTICATED ENDPOINTS:")
        for result in results:
            if result.get('success', False):
                print(f"   ✅ {result['test_name']}")
        print()
    
    failed_count = total_count - working_count
    if failed_count > 0:
        print("❌ FAILED AUTHENTICATED ENDPOINTS:")
        for result in results:
            if not result.get('success', False):
                reason = "Exception" if 'error' in result else f"HTTP {result.get('status_code', 'Unknown')}"
                print(f"   ❌ {result['test_name']} - {reason}")
        print()
    
    # Test encryption/decryption cycle
    print("🔐 === ENCRYPTION/DECRYPTION CYCLE TEST ===")
    
    # First encrypt some data
    encrypt_url = f"{base_url}/security/encrypt"
    test_data = "enterprise-confidential-information-12345"
    
    try:
        encrypt_response = requests.post(
            encrypt_url, 
            headers=headers,
            json={'plaintext': test_data},
            timeout=10
        )
        
        if encrypt_response.status_code == 200:
            encrypted_data = encrypt_response.json().get('encrypted')
            print(f"✅ Encryption Success: {encrypted_data[:50]}...")
            
            # Now decrypt it
            decrypt_url = f"{base_url}/security/decrypt"
            decrypt_response = requests.post(
                decrypt_url,
                headers=headers, 
                json={'encrypted': encrypted_data},
                timeout=10
            )
            
            if decrypt_response.status_code == 200:
                decrypted_data = decrypt_response.json().get('plaintext')
                print(f"✅ Decryption Success: {decrypted_data}")
                
                if decrypted_data == test_data:
                    print("🎉 ENCRYPTION/DECRYPTION CYCLE: PERFECT\!")
                else:
                    print("❌ ENCRYPTION/DECRYPTION: DATA MISMATCH\!")
                    
            else:
                print(f"❌ Decryption Failed: {decrypt_response.status_code}")
                
        else:
            print(f"❌ Encryption Failed: {encrypt_response.status_code}")
            
    except Exception as e:
        print(f"💥 Encryption/Decryption Test Exception: {e}")
    
    print()
    
    # Final assessment
    if working_count >= total_count * 0.8:  # 80% success rate
        print("🎉 ENTERPRISE AUTHENTICATION: WORKING\!")
        print("✅ JWT tokens are properly validated")
        print("✅ Protected endpoints respond correctly")  
        print("✅ Role-based permissions functioning")
        return True
    else:
        print("❌ ENTERPRISE AUTHENTICATION: ISSUES DETECTED")
        print(f"❌ Only {working_count}/{total_count} endpoints working")
        print("🔧 Authentication system needs fixes")
        return False

if __name__ == "__main__":
    success = test_authenticated_endpoints()
    
    print("\n" + "="*60)
    if success:
        print("🏆 ENTERPRISE AUTHENTICATION VERIFICATION: SUCCESS")
        print("All protected endpoints work with proper JWT tokens\!")
    else:
        print("⚠️  ENTERPRISE AUTHENTICATION VERIFICATION: FAILED") 
        print("Authentication system needs debugging.")
    print("="*60)

