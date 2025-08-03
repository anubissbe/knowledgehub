const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

async function detailedTest() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();
  
  console.log('🔍 Starting detailed UI analysis...\n');

  try {
    // Enable console logging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('❌ Console Error:', msg.text());
      }
    });

    // Monitor network requests
    const networkRequests = [];
    page.on('request', request => {
      if (request.url().includes('192.168.1.25')) {
        networkRequests.push({
          url: request.url(),
          method: request.method(),
          resourceType: request.resourceType()
        });
      }
    });

    page.on('response', response => {
      if (response.url().includes('192.168.1.25') && response.status() !== 200 && response.status() !== 304) {
        console.log(`⚠️ Non-200 response: ${response.status()} - ${response.url()}`);
      }
    });

    // Load the page
    console.log('1. Loading page...');
    await page.goto('http://192.168.1.25:3100', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    console.log('✅ Page loaded\n');

    // Check the DOM structure
    console.log('2. Analyzing DOM structure...');
    const rootElement = await page.$('#root');
    if (rootElement) {
      const rootHTML = await page.$eval('#root', el => el.innerHTML.substring(0, 500));
      console.log('Root element HTML preview:', rootHTML);
      console.log('Root element children count:', await page.$eval('#root', el => el.children.length));
    }

    // Check for React app
    const hasReactApp = await page.evaluate(() => {
      return window.React || window.__REACT_DEVTOOLS_GLOBAL_HOOK__ || document.querySelector('[data-reactroot]') !== null;
    });
    console.log('\n3. React app detected:', hasReactApp ? '✅ Yes' : '❌ No');

    // Set authentication
    console.log('\n4. Setting authentication...');
    await page.evaluate(() => {
      localStorage.setItem('knowledgehub_settings', JSON.stringify({
        apiUrl: "http://192.168.1.25:3000",
        apiKey: "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM",
        enableNotifications: true,
        autoRefresh: true,
        refreshInterval: 30,
        darkMode: false,
        language: "en",
        animationSpeed: 1,
        cacheSize: 100,
        maxMemories: 1000,
        compressionEnabled: true
      }));
    });
    await page.reload({ waitUntil: 'networkidle' });
    console.log('✅ Authentication set and page reloaded');

    // Wait a bit for any dynamic content
    await page.waitForTimeout(3000);

    // Check for common UI elements
    console.log('\n5. Checking for UI elements...');
    const elements = {
      'Navigation/Menu': ['nav', '.nav', '.navigation', '.menu', '.sidebar', '[role="navigation"]'],
      'Header': ['header', '.header', '[role="banner"]'],
      'Main Content': ['main', '.main', '[role="main"]', '.content'],
      'Links': ['a'],
      'Buttons': ['button', '[role="button"]'],
      'Cards/Features': ['.card', '.feature', '[class*="card"]', '[class*="feature"]']
    };

    for (const [name, selectors] of Object.entries(elements)) {
      for (const selector of selectors) {
        const count = await page.$$eval(selector, els => els.length);
        if (count > 0) {
          console.log(`✅ Found ${count} ${name} elements with selector: ${selector}`);
          break;
        }
      }
    }

    // Check for Material-UI components
    console.log('\n6. Checking for Material-UI components...');
    const muiClasses = await page.$$eval('[class*="MuiPaper"], [class*="MuiButton"], [class*="MuiTypography"], [class*="MuiGrid"]', 
      els => els.map(el => el.className).slice(0, 5)
    );
    console.log('Material-UI classes found:', muiClasses.length > 0 ? muiClasses : 'None');

    // Try to find any text content
    console.log('\n7. Looking for text content...');
    const bodyText = await page.$eval('body', el => el.innerText.substring(0, 500));
    console.log('Body text preview:', bodyText || '(empty)');

    // Check network requests
    console.log('\n8. Network requests summary:');
    console.log(`Total requests to 192.168.1.25: ${networkRequests.length}`);
    const requestTypes = {};
    networkRequests.forEach(req => {
      requestTypes[req.resourceType] = (requestTypes[req.resourceType] || 0) + 1;
    });
    console.log('Request types:', requestTypes);

    // Try different routes
    console.log('\n9. Testing specific routes...');
    const routes = ['/ai', '/memories', '/projects', '/analytics'];
    
    for (const route of routes) {
      await page.goto(`http://192.168.1.25:3100${route}`, { waitUntil: 'networkidle' });
      await page.waitForTimeout(1000);
      const hasContent = await page.$eval('body', el => el.innerText.length > 50);
      console.log(`Route ${route}: ${hasContent ? '✅ Has content' : '⚠️ Little/no content'}`);
    }

    // Final screenshot with DevTools info
    await page.screenshot({ 
      path: path.join(__dirname, 'knowledgehub-screenshots', 'detailed-analysis.png'),
      fullPage: true 
    });

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }
}

detailedTest().catch(console.error);