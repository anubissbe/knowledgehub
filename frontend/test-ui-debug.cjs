const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

async function debugUI() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();
  
  console.log('🔍 Debugging KnowledgeHub UI...\n');

  try {
    // Enable detailed console logging
    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();
      if (type === 'error') {
        console.log(`❌ Console Error: ${text}`);
      } else if (type === 'warning') {
        console.log(`⚠️ Console Warning: ${text}`);
      } else if (type === 'log' && (text.includes('API') || text.includes('Error'))) {
        console.log(`📝 Console Log: ${text}`);
      }
    });

    // Monitor failed requests
    page.on('requestfailed', request => {
      console.log(`❌ Request Failed: ${request.url()}`);
      console.log(`   Failure: ${request.failure()?.errorText}`);
    });

    // Load the page
    console.log('1. Loading page...');
    const response = await page.goto('http://192.168.1.25:3100', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    console.log(`✅ Page loaded with status: ${response.status()}\n`);

    // Set authentication
    console.log('2. Setting authentication...');
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

    // Force a reload to pick up the authentication
    await page.reload({ waitUntil: 'networkidle' });
    console.log('✅ Authentication set and page reloaded\n');

    // Wait for React to render
    await page.waitForTimeout(3000);

    // Debug DOM structure
    console.log('3. Analyzing DOM structure...');
    
    // Check if React rendered
    const reactRoot = await page.$eval('#root', el => {
      return {
        hasChildren: el.children.length > 0,
        innerHTML: el.innerHTML.substring(0, 200),
        childCount: el.children.length,
        firstChildTag: el.firstElementChild?.tagName
      };
    });
    console.log('React root:', JSON.stringify(reactRoot, null, 2));

    // Look for MUI components
    const muiComponents = await page.$$eval('[class*="Mui"]', els => {
      const components = {};
      els.forEach(el => {
        const classes = el.className.split(' ');
        classes.forEach(cls => {
          if (cls.startsWith('Mui')) {
            const component = cls.split('-')[0];
            components[component] = (components[component] || 0) + 1;
          }
        });
      });
      return components;
    });
    console.log('\nMUI Components found:', muiComponents);

    // Check for specific elements
    console.log('\n4. Looking for specific UI elements...');
    
    const selectors = {
      'App Bar': '.MuiAppBar-root',
      'Drawer/Sidebar': '.MuiDrawer-root',
      'Navigation List': '.MuiList-root',
      'Main Content': 'main',
      'Dashboard Cards': '.MuiCard-root',
      'Typography': '.MuiTypography-root'
    };

    for (const [name, selector] of Object.entries(selectors)) {
      const count = await page.$$eval(selector, els => els.length);
      const visible = count > 0 ? await page.$eval(selector, el => {
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }) : false;
      console.log(`${name}: ${count} found, ${visible ? 'visible' : 'not visible'}`);
    }

    // Check actual content
    console.log('\n5. Checking page content...');
    const pageText = await page.$eval('body', el => el.innerText);
    console.log(`Total text length: ${pageText.length} characters`);
    console.log('First 300 chars of content:');
    console.log(pageText.substring(0, 300));

    // Check for navigation items
    console.log('\n6. Looking for navigation items...');
    const navItems = await page.$$eval('a, button', els => {
      return els
        .filter(el => el.textContent.trim().length > 0)
        .map(el => ({
          tag: el.tagName,
          text: el.textContent.trim(),
          href: el.href || null,
          classes: el.className
        }))
        .slice(0, 10); // First 10 items
    });
    console.log('Navigation items found:', navItems);

    // Screenshot
    await page.screenshot({ 
      path: path.join(__dirname, 'knowledgehub-screenshots', 'debug-screenshot.png'),
      fullPage: true 
    });
    console.log('\n✅ Debug screenshot saved');

  } catch (error) {
    console.error('Test error:', error);
  } finally {
    await browser.close();
  }
}

debugUI().catch(console.error);