// LAN Configuration Script for KnowledgeHub Web UI
// Run this in your browser console when accessing from http://192.168.1.25:3100

const LAN_SETTINGS = {
  apiUrl: 'http://192.168.1.25:3000',
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

console.log("=== KnowledgeHub LAN Setup ===");
console.log("Setting up API configuration for LAN access...");

// Set the configuration
localStorage.setItem('knowledgehub_settings', JSON.stringify(LAN_SETTINGS));

console.log("✅ Configuration saved!");
console.log("API URL:", LAN_SETTINGS.apiUrl);
console.log("API Key:", LAN_SETTINGS.apiKey.substring(0, 20) + "...");

console.log("\n📋 Next steps:");
console.log("1. Refresh the page (F5)");
console.log("2. Navigate to 'AI Intelligence' from the menu");
console.log("3. You should see all 8 AI features with live data");

// Test API connectivity
console.log("\n🔧 Testing API connection...");
fetch(LAN_SETTINGS.apiUrl + '/health', {
  headers: {
    'X-API-Key': LAN_SETTINGS.apiKey
  }
})
.then(response => response.json())
.then(data => {
  console.log("✅ API is reachable:", data);
})
.catch(error => {
  console.error("❌ API connection failed:", error);
  console.log("Make sure the API is running on http://192.168.1.25:3000");
});

// For easy copy-paste:
console.log("\n📋 Quick copy-paste command:");
console.log(`localStorage.setItem('knowledgehub_settings', '${JSON.stringify(LAN_SETTINGS)}')`);