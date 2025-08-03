// Script to set the API key in localStorage for the frontend
const settings = {
  apiUrl: 'http://localhost:3000',
  apiKey: 'knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM',
  enableNotifications: true,
  autoRefresh: true,
  refreshInterval: 30,
  darkMode: false,
  language: 'en',
  animationSpeed: 1,
  cacheSize: 100,
  maxMemories: 1000,
  compressionEnabled: true
};

// This would be run in the browser console
console.log("Run this in your browser console at http://localhost:3100:");
console.log(`localStorage.setItem('knowledgehub_settings', '${JSON.stringify(settings)}');`);
console.log("Then refresh the page.");

// Alternative: Create a curl command to test the API
console.log("\nOr test the API directly:");
console.log(`curl -H "X-API-Key: ${settings.apiKey}" http://localhost:3000/api/monitoring/ai-features/status | jq`);