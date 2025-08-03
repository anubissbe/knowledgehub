import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 }
  });
  const page = await context.newPage();

  // Log console messages
  page.on('console', msg => {
    if (msg.type() === 'error' || msg.type() === 'warning') {
      console.log(`Console ${msg.type()}: ${msg.text()}`);
    }
  });

  // Navigate to the app
  console.log('Navigating to http://192.168.1.25:3101');
  const response = await page.goto('http://192.168.1.25:3101', { waitUntil: 'networkidle' });
  console.log(`Initial response status: ${response.status()}`);
  
  // Set authentication in localStorage
  await page.evaluate(() => {
    localStorage.setItem('knowledgehub_settings', '{"apiUrl":"http://192.168.1.25:3000","apiKey":"knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM","enableNotifications":true,"autoRefresh":true,"refreshInterval":30,"darkMode":false,"language":"en","animationSpeed":1,"cacheSize":100,"maxMemories":1000,"compressionEnabled":true}');
  });
  
  // Reload to apply settings
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
  
  // Take screenshot of home page
  await page.screenshot({ path: 'home-spa.png', fullPage: true });
  console.log('Captured home page screenshot');
  
  // Check if navigation menu exists
  const navLinks = await page.$$('nav a, [role="navigation"] a, header a');
  console.log(`Found ${navLinks.length} navigation links`);
  
  // Test 1: Navigate to AI Intelligence using client-side routing
  console.log('\n=== Testing AI Intelligence Page (Client-side navigation) ===');
  
  // Try to click on AI Intelligence link
  const aiLink = await page.$('a[href="/ai-intelligence"], a[href="#/ai-intelligence"], a:has-text("AI Intelligence"), a:has-text("AI")');
  if (aiLink) {
    await aiLink.click();
    await page.waitForTimeout(3000);
    console.log('Clicked AI Intelligence link');
  } else {
    console.log('AI Intelligence link not found, trying navigation menu');
    // Try to find in a dropdown or menu
    const menuButton = await page.$('button:has-text("Menu"), [aria-label*="menu"], .menu-button');
    if (menuButton) {
      await menuButton.click();
      await page.waitForTimeout(500);
    }
  }
  
  const currentUrl = page.url();
  console.log(`Current URL: ${currentUrl}`);
  
  await page.screenshot({ path: 'ai-intelligence-spa.png', fullPage: true });
  
  // Check for AI features
  const features = await page.$$('[class*="feature"], [class*="Feature"], .card, .MuiCard-root');
  console.log(`Found ${features.length} potential feature elements`);
  
  // Test 2: Navigate to Memory System
  console.log('\n=== Testing Memory System Page (Client-side navigation) ===');
  await page.goto('http://192.168.1.25:3101', { waitUntil: 'networkidle' });
  
  const memoryLink = await page.$('a[href="/memory"], a[href="#/memory"], a:has-text("Memory")');
  if (memoryLink) {
    await memoryLink.click();
    await page.waitForTimeout(3000);
    console.log('Clicked Memory link');
  }
  
  await page.screenshot({ path: 'memory-spa.png', fullPage: true });
  
  // Check for memory stats
  const stats = await page.$$('[class*="stat"], [class*="Stat"], .MuiPaper-root');
  console.log(`Found ${stats.length} potential stat elements`);
  
  // Test 3: Check React app structure
  console.log('\n=== Checking React App Structure ===');
  const appRoot = await page.$('#root, .App, [data-reactroot]');
  if (appRoot) {
    console.log('React app root found');
    const rootHtml = await appRoot.innerHTML();
    console.log(`Root element has content: ${rootHtml.length > 100}`);
  } else {
    console.log('React app root not found');
  }
  
  // Check if the app loaded properly
  const bodyText = await page.textContent('body');
  if (bodyText.includes('Loading') || bodyText.includes('loading')) {
    console.log('App appears to be stuck in loading state');
  }
  
  await browser.close();
})();