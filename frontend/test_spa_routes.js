import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 }
  });
  const page = await context.newPage();

  // Set authentication before navigation
  await context.addInitScript(() => {
    localStorage.setItem('knowledgehub_settings', '{"apiUrl":"http://192.168.1.25:3000","apiKey":"knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM","enableNotifications":true,"autoRefresh":true,"refreshInterval":30,"darkMode":false,"language":"en","animationSpeed":1,"cacheSize":100,"maxMemories":1000,"compressionEnabled":true}');
  });

  const testRoutes = [
    { path: '/', name: 'Home/Dashboard', selector: '.dashboard, .home, #root' },
    { path: '/ai-intelligence', name: 'AI Intelligence', selector: '.ai-intelligence, .feature-card, [class*="AIIntelligence"]' },
    { path: '/memory', name: 'Memory System', selector: '.memory-system, .stats-container, [class*="Memory"]' },
    { path: '/sources', name: 'Sources', selector: '.sources, .sources-list, [class*="Sources"]' },
    { path: '/settings', name: 'Settings', selector: '.settings, .settings-form, [class*="Settings"]' }
  ];

  console.log('=== Testing SPA Routes with Production Build ===\n');

  for (const route of testRoutes) {
    console.log(`\nTesting ${route.name} (${route.path})...`);
    
    // Navigate directly to the route
    const response = await page.goto(`http://192.168.1.25:3101${route.path}`, { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    console.log(`Response status: ${response.status()}`);
    
    // Wait for React to render
    await page.waitForTimeout(2000);
    
    // Check if the page loaded
    const title = await page.title();
    console.log(`Page title: ${title}`);
    
    // Check for route-specific content
    const elements = await page.$$(route.selector);
    console.log(`Found ${elements.length} elements matching selector: ${route.selector}`);
    
    // Check for any content
    const bodyText = await page.textContent('body');
    const hasContent = bodyText && bodyText.trim().length > 100;
    console.log(`Page has content: ${hasContent}`);
    
    // Take screenshot
    const screenshotName = `${route.name.toLowerCase().replace(/[^a-z0-9]/g, '-')}-spa.png`;
    await page.screenshot({ path: screenshotName, fullPage: true });
    console.log(`Screenshot saved: ${screenshotName}`);
    
    // Check for React rendering
    const reactRoot = await page.$('#root');
    if (reactRoot) {
      const rootClasses = await reactRoot.getAttribute('class');
      console.log(`React root classes: ${rootClasses || 'none'}`);
    }
  }

  // Test client-side navigation
  console.log('\n\n=== Testing Client-Side Navigation ===\n');
  
  // Go to home first
  await page.goto('http://192.168.1.25:3101/', { waitUntil: 'networkidle' });
  
  // Look for navigation elements
  const navSelectors = [
    'nav a',
    '[role="navigation"] a',
    'header a',
    '.MuiDrawer-root a',
    '.sidebar a',
    '[class*="nav"] a',
    '[class*="Nav"] a'
  ];
  
  let navLinks = [];
  for (const selector of navSelectors) {
    const links = await page.$$(selector);
    if (links.length > 0) {
      console.log(`Found ${links.length} links with selector: ${selector}`);
      navLinks = links;
      break;
    }
  }
  
  if (navLinks.length > 0) {
    console.log(`\nTotal navigation links found: ${navLinks.length}`);
    
    // Get link details
    for (let i = 0; i < Math.min(navLinks.length, 5); i++) {
      const text = await navLinks[i].textContent();
      const href = await navLinks[i].getAttribute('href');
      console.log(`Link ${i + 1}: "${text}" -> ${href}`);
    }
  } else {
    console.log('No navigation links found - app may be using a different navigation pattern');
  }

  await browser.close();
  
  console.log('\n=== Test Complete ===');
})();