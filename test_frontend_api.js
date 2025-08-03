// Test script to verify frontend can access API with authentication

const API_KEY = 'knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM';
const BASE_URL = 'http://localhost:3000';

// Endpoints to test
const endpoints = [
  '/api/claude-auto/memory/stats',
  '/api/mistake-learning/patterns',
  '/api/proactive/health',
  '/api/decisions/search?query=test',
  '/api/performance/stats',
  '/api/monitoring/ai-features/status'
];

async function testEndpoint(endpoint) {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      headers: {
        'X-API-Key': API_KEY,
        'Content-Type': 'application/json'
      }
    });
    
    const data = await response.json();
    console.log(`✅ ${endpoint} - Status: ${response.status}`);
    return true;
  } catch (error) {
    console.log(`❌ ${endpoint} - Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log('Testing Frontend API Access...\n');
  
  let passed = 0;
  for (const endpoint of endpoints) {
    if (await testEndpoint(endpoint)) {
      passed++;
    }
  }
  
  console.log(`\nResults: ${passed}/${endpoints.length} endpoints accessible`);
  
  // Instructions for setting up the frontend
  console.log('\n📋 To configure the frontend:');
  console.log('1. Open http://localhost:3100 in your browser');
  console.log('2. Open browser console (F12)');
  console.log('3. Run this command:');
  console.log(`localStorage.setItem('knowledgehub_settings', '${JSON.stringify({
    apiUrl: 'http://localhost:3000',
    apiKey: API_KEY,
    enableNotifications: true,
    autoRefresh: true,
    refreshInterval: 30,
    darkMode: false,
    language: 'en',
    animationSpeed: 1,
    cacheSize: 100,
    maxMemories: 1000,
    compressionEnabled: true
  })}')`);
  console.log('4. Refresh the page');
  console.log('5. Navigate to the AI Intelligence page');
}

// Note: This script requires Node.js 18+ for native fetch
// Run with: node test_frontend_api.js
if (typeof window === 'undefined') {
  main();
}