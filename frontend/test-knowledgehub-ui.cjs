const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

async function testKnowledgeHub() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }
  });
  const page = await context.newPage();
  
  // Create screenshots directory
  const screenshotDir = path.join(__dirname, 'knowledgehub-screenshots');
  if (!fs.existsSync(screenshotDir)) {
    fs.mkdirSync(screenshotDir);
  }

  const results = {
    timestamp: new Date().toISOString(),
    tests: []
  };

  try {
    console.log('🚀 Starting KnowledgeHub UI tests...\n');

    // Test 1: Basic accessibility
    console.log('1. Testing basic accessibility via LAN IP...');
    let loadSuccess = false;
    try {
      await page.goto('http://192.168.1.25:3100', { 
        waitUntil: 'networkidle',
        timeout: 30000 
      });
      loadSuccess = true;
      results.tests.push({
        test: 'Basic accessibility',
        status: 'PASS',
        details: 'Page loaded successfully via LAN IP'
      });
      console.log('✅ Page loaded successfully');
    } catch (error) {
      results.tests.push({
        test: 'Basic accessibility',
        status: 'FAIL',
        details: error.message
      });
      console.log('❌ Failed to load page:', error.message);
    }

    if (loadSuccess) {
      // Set authentication in localStorage
      console.log('\n2. Setting authentication...');
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
      console.log('✅ Authentication set');

      // Take initial screenshot
      await page.screenshot({ 
        path: path.join(screenshotDir, '01-homepage.png'),
        fullPage: true 
      });

      // Test 3: Check navigation sidebar
      console.log('\n3. Checking navigation sidebar...');
      const sidebar = await page.$('.sidebar, nav, [class*="sidebar"], [class*="navigation"], [class*="menu"]');
      if (sidebar) {
        const isVisible = await sidebar.isVisible();
        results.tests.push({
          test: 'Navigation sidebar',
          status: isVisible ? 'PASS' : 'FAIL',
          details: isVisible ? 'Sidebar is visible' : 'Sidebar exists but not visible'
        });
        console.log(isVisible ? '✅ Sidebar is visible' : '⚠️ Sidebar exists but not visible');
      } else {
        results.tests.push({
          test: 'Navigation sidebar',
          status: 'FAIL',
          details: 'No sidebar found'
        });
        console.log('❌ No sidebar found');
      }

      // Test 4: Check for API calls
      console.log('\n4. Monitoring API calls...');
      const apiCalls = [];
      const notFoundCalls = [];
      
      page.on('response', response => {
        const url = response.url();
        if (url.includes('192.168.1.25:3000')) {
          const status = response.status();
          apiCalls.push({ url, status });
          if (status === 404) {
            notFoundCalls.push(url);
          }
        }
      });

      // Navigate to different pages
      console.log('\n5. Testing page navigation...');
      
      // Try to find and click on AI Intelligence link
      const aiLink = await page.$('a[href*="ai"], button:has-text("AI"), [class*="ai"]:has-text("Intelligence")');
      if (aiLink) {
        await aiLink.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ 
          path: path.join(screenshotDir, '02-ai-intelligence.png'),
          fullPage: true 
        });
        console.log('✅ Navigated to AI Intelligence page');
      }

      // Try memories page
      await page.goto('http://192.168.1.25:3100/memories', { waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      await page.screenshot({ 
        path: path.join(screenshotDir, '03-memories.png'),
        fullPage: true 
      });

      // Try projects page
      await page.goto('http://192.168.1.25:3100/projects', { waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      await page.screenshot({ 
        path: path.join(screenshotDir, '04-projects.png'),
        fullPage: true 
      });

      // Try analytics page
      await page.goto('http://192.168.1.25:3100/analytics', { waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      await page.screenshot({ 
        path: path.join(screenshotDir, '05-analytics.png'),
        fullPage: true 
      });

      // Analyze API calls
      results.tests.push({
        test: 'API calls',
        status: notFoundCalls.length === 0 ? 'PASS' : 'PARTIAL',
        details: {
          totalCalls: apiCalls.length,
          notFoundCalls: notFoundCalls.length,
          notFoundUrls: notFoundCalls
        }
      });
      console.log(`\n📊 API Call Summary:`);
      console.log(`   Total API calls: ${apiCalls.length}`);
      console.log(`   404 errors: ${notFoundCalls.length}`);
      
      // Test 6: Check AI Intelligence features
      console.log('\n6. Checking AI Intelligence features...');
      await page.goto('http://192.168.1.25:3100/ai', { waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      
      const aiFeatures = await page.$$eval('[class*="feature"], [class*="card"], [class*="ai-"]', elements => 
        elements.map(el => ({
          text: el.textContent,
          classes: el.className
        }))
      );
      
      results.tests.push({
        test: 'AI Intelligence features',
        status: aiFeatures.length > 0 ? 'PASS' : 'FAIL',
        details: `Found ${aiFeatures.length} AI feature elements`
      });
      console.log(`✅ Found ${aiFeatures.length} AI feature elements`);

      // Take final screenshot
      await page.screenshot({ 
        path: path.join(screenshotDir, '06-final-state.png'),
        fullPage: true 
      });

      // Test 7: Check console errors
      const consoleErrors = [];
      page.on('console', msg => {
        if (msg.type() === 'error') {
          consoleErrors.push(msg.text());
        }
      });
      
      await page.reload({ waitUntil: 'networkidle' });
      await page.waitForTimeout(2000);
      
      results.tests.push({
        test: 'Console errors',
        status: consoleErrors.length === 0 ? 'PASS' : 'WARN',
        details: {
          errorCount: consoleErrors.length,
          errors: consoleErrors.slice(0, 5) // First 5 errors
        }
      });
    }

  } catch (error) {
    console.error('Test execution error:', error);
    results.error = error.message;
  } finally {
    await browser.close();
  }

  // Save results
  fs.writeFileSync(
    path.join(screenshotDir, 'test-results.json'),
    JSON.stringify(results, null, 2)
  );

  // Print summary
  console.log('\n📋 Test Summary:');
  console.log('================');
  results.tests.forEach(test => {
    const icon = test.status === 'PASS' ? '✅' : test.status === 'WARN' ? '⚠️' : '❌';
    console.log(`${icon} ${test.test}: ${test.status}`);
    if (test.details && typeof test.details === 'object') {
      console.log(`   Details:`, JSON.stringify(test.details, null, 2));
    } else if (test.details) {
      console.log(`   Details: ${test.details}`);
    }
  });

  return results;
}

// Run the test
testKnowledgeHub().catch(console.error);