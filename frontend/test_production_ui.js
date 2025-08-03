import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 }
  });
  const page = await context.newPage();

  // Log console messages
  page.on('console', msg => {
    console.log(`Console ${msg.type()}: ${msg.text()}`);
  });

  // Log network errors
  page.on('pageerror', error => {
    console.error('Page error:', error.message);
  });

  // Navigate to the app
  console.log('Navigating to http://192.168.1.25:3100');
  const response = await page.goto('http://192.168.1.25:3100', { waitUntil: 'networkidle' });
  console.log(`Initial response status: ${response.status()}`);
  
  // Set authentication in localStorage
  await page.evaluate(() => {
    localStorage.setItem('knowledgehub_settings', '{"apiUrl":"http://192.168.1.25:3000","apiKey":"knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM","enableNotifications":true,"autoRefresh":true,"refreshInterval":30,"darkMode":false,"language":"en","animationSpeed":1,"cacheSize":100,"maxMemories":1000,"compressionEnabled":true}');
  });
  
  // Reload to apply settings
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  
  // Take screenshot of initial page
  await page.screenshot({ path: 'home-production.png', fullPage: true });
  console.log('Captured home page screenshot');

  // Test 1: AI Intelligence page
  console.log('\n=== Testing AI Intelligence Page ===');
  const aiResponse = await page.goto('http://192.168.1.25:3100/ai-intelligence', { waitUntil: 'networkidle' });
  console.log(`AI Intelligence page status: ${aiResponse.status()}`);
  await page.waitForTimeout(3000);
  
  // Check the page content
  const pageContent = await page.content();
  console.log(`Page has content: ${pageContent.length > 1000}`);
  
  await page.screenshot({ path: 'ai-intelligence-production.png', fullPage: true });
  
  // Check if feature cards are visible with various selectors
  const featureCards = await page.$$('.feature-card');
  const aiFeatures = await page.$$('[class*="feature"]');
  const cards = await page.$$('[class*="card"]');
  console.log(`Found ${featureCards.length} .feature-card elements`);
  console.log(`Found ${aiFeatures.length} elements with "feature" in class`);
  console.log(`Found ${cards.length} elements with "card" in class`);
  
  // Test 2: Memory System page
  console.log('\n=== Testing Memory System Page ===');
  await page.goto('http://192.168.1.25:3100/memory');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'memory-system-production.png', fullPage: true });
  
  // Check if stats are visible
  const statsVisible = await page.isVisible('.stats-container');
  console.log(`Stats container visible: ${statsVisible}`);
  
  // Test 3: Sources page
  console.log('\n=== Testing Sources Page ===');
  await page.goto('http://192.168.1.25:3100/sources');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'sources-production.png', fullPage: true });
  
  // Check if sources list is visible
  const sourcesVisible = await page.isVisible('.sources-list');
  console.log(`Sources list visible: ${sourcesVisible}`);
  
  // Test 4: Settings page
  console.log('\n=== Testing Settings Page ===');
  await page.goto('http://192.168.1.25:3100/settings');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'settings-production.png', fullPage: true });
  
  // Check if settings form is visible
  const settingsForm = await page.isVisible('.settings-form');
  console.log(`Settings form visible: ${settingsForm}`);
  
  // Check for any console errors
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.error('Console error:', msg.text());
    }
  });

  await browser.close();
})();