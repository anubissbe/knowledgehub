const { chromium } = require('@playwright/test');

async function testKnowledgeHubUI() {
  console.log('\n🚀 Starting KnowledgeHub UI Validation...\n');
  console.log('📍 Testing URL: http://localhost:3001\n');

  const browser = await chromium.launch({ 
    headless: true,
    slowMo: 100
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    ignoreHTTPSErrors: true
  });
  
  const page = await context.newPage();
  
  // Track errors
  const consoleErrors = [];
  const pageErrors = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });
  
  page.on('pageerror', err => {
    pageErrors.push(err.message);
  });

  const results = {
    tests: [],
    errors: []
  };

  try {
    // Navigate to the application
    await page.goto('http://localhost:3001', { waitUntil: 'domcontentloaded', timeout: 10000 });
    results.tests.push('✅ Application loaded successfully');
    
    // Set up authentication
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
    results.tests.push('✅ Authentication configured');
    
    // Reload to apply settings
    await page.reload({ waitUntil: 'domcontentloaded' });
    
    // Take screenshots and test pages
    const pages = [
      { name: 'Dashboard', path: '/', screenshot: '01_dashboard.png' },
      { name: 'AI Intelligence', path: '/ai', screenshot: '02_ai.png' },
      { name: 'Memory System', path: '/memory', screenshot: '03_memory.png' },
      { name: 'Search Knowledge', path: '/search', screenshot: '04_search.png' },
      { name: 'Sources', path: '/sources', screenshot: '05_sources.png' },
      { name: 'Settings', path: '/settings', screenshot: '06_settings.png' }
    ];
    
    for (const pageInfo of pages) {
      try {
        await page.goto(`http://localhost:3001${pageInfo.path}`, { 
          waitUntil: 'domcontentloaded',
          timeout: 10000 
        });
        await page.waitForTimeout(1000); // Wait for content to load
        await page.screenshot({ 
          path: `screenshots/${pageInfo.screenshot}`, 
          fullPage: true 
        });
        results.tests.push(`✅ ${pageInfo.name} page loaded`);
      } catch (error) {
        results.errors.push(`❌ ${pageInfo.name} page failed: ${error.message}`);
      }
    }
    
    // Check for console errors
    if (consoleErrors.length > 0) {
      results.errors.push(`❌ Console errors found: ${consoleErrors.length}`);
      consoleErrors.forEach(err => console.log('  Console Error:', err));
    } else {
      results.tests.push('✅ No console errors');
    }
    
    // Check for page errors
    if (pageErrors.length > 0) {
      results.errors.push(`❌ Page errors found: ${pageErrors.length}`);
      pageErrors.forEach(err => console.log('  Page Error:', err));
    } else {
      results.tests.push('✅ No page errors');
    }
    
  } catch (error) {
    results.errors.push(`❌ Critical error: ${error.message}`);
  } finally {
    await browser.close();
  }
  
  // Print results
  console.log('\n📊 VALIDATION RESULTS\n' + '='.repeat(50));
  
  console.log('\n✅ Successful Tests:');
  results.tests.forEach(test => console.log(`  ${test}`));
  
  if (results.errors.length > 0) {
    console.log('\n❌ Errors:');
    results.errors.forEach(error => console.log(`  ${error}`));
  }
  
  const score = (results.tests.length / (results.tests.length + results.errors.length)) * 100;
  console.log('\n' + '='.repeat(50));
  console.log(`\n🎯 OVERALL SYSTEM HEALTH SCORE: ${score.toFixed(1)}%\n`);
  
  return results;
}

// Run the test
testKnowledgeHubUI().catch(console.error);