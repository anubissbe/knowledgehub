#!/usr/bin/env python3
import requests

# Test with minimal headers
url = "http://localhost:3000/health"

# Test 1: Default requests
print("Test 1: Default requests")
r1 = requests.get(url)
print(f"Status: {r1.status_code}")
print(f"Response: {r1.text[:100]}...")

# Test 2: With curl user agent
print("\nTest 2: With curl user agent")
r2 = requests.get(url, headers={"User-Agent": "curl/7.81.0"})
print(f"Status: {r2.status_code}")
print(f"Response: {r2.text[:100]}...")

# Test 3: Try different paths
print("\nTest 3: API endpoint")
r3 = requests.get("http://localhost:3000/api/claude-auto/memory/stats")
print(f"Status: {r3.status_code}")

# Test 4: With all curl headers
print("\nTest 4: With all curl headers")
curl_headers = {
    "Host": "localhost:3000",
    "User-Agent": "curl/7.81.0",
    "Accept": "*/*"
}
r4 = requests.get(url, headers=curl_headers)
print(f"Status: {r4.status_code}")
print(f"Response: {r4.text[:100]}...")