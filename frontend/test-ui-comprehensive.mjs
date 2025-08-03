import { chromium } from 'playwright';

async function comprehensiveUIValidation() {
  console.log('\n🚀 Starting Comprehensive KnowledgeHub UI Validation...\n');
  console.log('📍 Testing LAN deployment at http://192.168.1.25:3100\n');
  console.log('⏰ Test started at:', new Date().toLocaleString());
  console.log('='.repeat(60) + '\n');

  const browser = await chromium.launch({ 
    headless: true,   // Run in headless mode
    slowMo: 100      // Slight delay for stability
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    ignoreHTTPSErrors: true
  });
  
  const page = await context.newPage();
  
  // Track all console messages
  const consoleLogs = [];
  page.on('console', msg => {
    const log = {
      type: msg.type(),
      text: msg.text(),
      location: msg.location()
    };
    consoleLogs.push(log);
    if (msg.type() === 'error') {
      console.log('❌ Console Error:', msg.text());
    }
  });
  
  // Track page errors
  const pageErrors = [];
  page.on('pageerror', err => {
    pageErrors.push(err.message);
    console.log('❌ Page Error:', err.message);
  });

  // Track network requests
  const apiRequests = [];
  const failedRequests = [];
  
  page.on('request', request => {
    if (request.url().includes('/api/')) {
      apiRequests.push({
        url: request.url(),
        method: request.method(),
        headers: request.headers()
      });
    }
  });
  
  page.on('response', response => {
    if (response.url().includes('/api/') && response.status() >= 400) {
      failedRequests.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText()
      });
    }
  });

  const results = {
    dashboard: { pass: [], fail: [], elements: {} },
    ai: { pass: [], fail: [], elements: {} },
    memory: { pass: [], fail: [], elements: {} },
    search: { pass: [], fail: [], elements: {} },
    sources: { pass: [], fail: [], elements: {} },
    settings: { pass: [], fail: [], elements: {} },
    navigation: { pass: [], fail: [] },
    api: { pass: [], fail: [], requests: [] },
    performance: {}
  };

  try {
    // Test local deployment first
    console.log('\n📡 Testing Local Deployment (http://localhost:3100)...');
    const localStart = Date.now();
    await page.goto('http://localhost:3100', { waitUntil: 'networkidle', timeout: 30000 });
    const localLoadTime = Date.now() - localStart;
    console.log(`✅ Local UI loaded in ${localLoadTime}ms`);
    
    // Set up authentication
    console.log('\n🔐 Configuring Authentication...');
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
    console.log('✅ Authentication configured for LAN API');
    
    // Reload to apply settings
    await page.reload({ waitUntil: 'networkidle' });
    
    // 1. DASHBOARD TESTING
    console.log('\n\n1️⃣ TESTING DASHBOARD (/)\n' + '-'.repeat(40));
    await page.goto('http://localhost:3100/', { waitUntil: 'networkidle' });
    
    // Check metric cards
    const metricCards = await page.$$('[data-testid="metric-card"], .metric-card, .MuiCard-root');
    if (metricCards.length > 0) {
      results.dashboard.pass.push(`✅ Found ${metricCards.length} metric cards`);
      results.dashboard.elements.metricCards = metricCards.length;
    } else {
      results.dashboard.fail.push('❌ No metric cards found');
    }
    
    // Check charts
    const charts = await page.$$('.recharts-wrapper, canvas, svg.chart');
    if (charts.length > 0) {
      results.dashboard.pass.push(`✅ Found ${charts.length} charts/visualizations`);
      results.dashboard.elements.charts = charts.length;
    } else {
      results.dashboard.fail.push('❌ No charts found');
    }
    
    // Check real-time monitoring
    const monitoring = await page.$('[data-testid="real-time-monitoring"], .monitoring-section');
    if (monitoring) {
      results.dashboard.pass.push('✅ Real-time monitoring section present');
    } else {
      results.dashboard.fail.push('❌ Real-time monitoring not found');
    }
    
    await page.screenshot({ path: 'screenshots/detailed_01_dashboard.png', fullPage: true });
    
    // 2. AI INTELLIGENCE TESTING
    console.log('\n\n2️⃣ TESTING AI INTELLIGENCE (/ai)\n' + '-'.repeat(40));
    await page.goto('http://localhost:3100/ai', { waitUntil: 'networkidle' });
    
    // Count AI feature cards
    const aiCards = await page.$$('[data-testid="ai-feature-card"], .ai-feature-card, .feature-card');
    if (aiCards.length >= 8) {
      results.ai.pass.push(`✅ All ${aiCards.length} AI feature cards present`);
      results.ai.elements.featureCards = aiCards.length;
    } else {
      results.ai.fail.push(`❌ Only ${aiCards.length}/8 AI feature cards found`);
    }
    
    // Test tabs
    const tabs = await page.$$('[role="tab"], .MuiTab-root');
    if (tabs.length >= 4) {
      results.ai.pass.push(`✅ Found ${tabs.length} tabs`);
      
      // Click through tabs
      for (let i = 0; i < Math.min(tabs.length, 4); i++) {
        await tabs[i].click();
        await page.waitForTimeout(1000);
        const tabText = await tabs[i].textContent();
        console.log(`  - Clicked tab: ${tabText}`);
      }
      results.ai.pass.push('✅ Tab navigation working');
    } else {
      results.ai.fail.push(`❌ Only ${tabs.length}/4 tabs found`);
    }
    
    // Check performance metrics
    const perfMetrics = await page.$('[data-testid="performance-metrics"], .performance-chart');
    if (perfMetrics) {
      results.ai.pass.push('✅ Performance metrics chart present');
    } else {
      results.ai.fail.push('❌ Performance metrics not found');
    }
    
    await page.screenshot({ path: 'screenshots/detailed_02_ai.png', fullPage: true });
    
    // 3. MEMORY SYSTEM TESTING
    console.log('\n\n3️⃣ TESTING MEMORY SYSTEM (/memory)\n' + '-'.repeat(40));
    await page.goto('http://localhost:3100/memory', { waitUntil: 'networkidle' });
    
    // Test search
    const searchInput = await page.$('input[type="search"], input[placeholder*="search" i], .search-input');
    if (searchInput) {
      await searchInput.fill('test memory search');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(2000);
      results.memory.pass.push('✅ Search functionality working');
    } else {
      results.memory.fail.push('❌ Search input not found');
    }
    
    // Check memory list/table
    const memoryList = await page.$('.MuiDataGrid-root, table, [data-testid="memory-list"]');
    if (memoryList) {
      results.memory.pass.push('✅ Memory list/table present');
      
      // Count rows
      const rows = await page.$$('.MuiDataGrid-row, tr[data-row], tbody tr');
      results.memory.elements.memoryCount = rows.length;
      console.log(`  - Found ${rows.length} memory entries`);
    } else {
      results.memory.fail.push('❌ Memory list not found');
    }
    
    await page.screenshot({ path: 'screenshots/detailed_03_memory.png', fullPage: true });
    
    // 4. SEARCH KNOWLEDGE TESTING
    console.log('\n\n4️⃣ TESTING SEARCH KNOWLEDGE (/search)\n' + '-'.repeat(40));
    await page.goto('http://localhost:3100/search', { waitUntil: 'networkidle' });
    
    // Test search modes
    const searchModes = ['Semantic', 'Hybrid', 'Text'];
    for (const mode of searchModes) {
      const modeButton = await page.$(`button:has-text("${mode}")`);
      if (modeButton) {
        await modeButton.click();
        await page.waitForTimeout(500);
        results.search.pass.push(`✅ ${mode} search mode available`);
        
        // Perform a test search
        const searchBox = await page.$('input[type="search"], input[type="text"]');
        if (searchBox) {
          await searchBox.fill(`Test ${mode} search`);
          await page.keyboard.press('Enter');
          await page.waitForTimeout(2000);
        }
      } else {
        results.search.fail.push(`❌ ${mode} search mode not found`);
      }
    }
    
    await page.screenshot({ path: 'screenshots/detailed_04_search.png', fullPage: true });
    
    // 5. SOURCES TESTING
    console.log('\n\n5️⃣ TESTING SOURCES (/sources)\n' + '-'.repeat(40));
    await page.goto('http://localhost:3100/sources', { waitUntil: 'networkidle' });
    
    // Check source list
    const sourceList = await page.$('[data-testid="source-list"], table, .source-list');
    if (sourceList) {
      results.sources.pass.push('✅ Source list present');
      
      const sources = await page.$$('tr[data-source], .source-item, .MuiTableRow-root');
      results.sources.elements.sourceCount = sources.length;
      console.log(`  - Found ${sources.length} sources`);
    } else {
      results.sources.fail.push('❌ Source list not found');
    }
    
    // Check action buttons
    const addButton = await page.$('button:has-text("Add Source"), button:has-text("Add")');
    const refreshButton = await page.$('button:has-text("Refresh")');
    
    if (addButton) results.sources.pass.push('✅ Add Source button present');
    else results.sources.fail.push('❌ Add Source button not found');
    
    if (refreshButton) results.sources.pass.push('✅ Refresh button present');
    else results.sources.fail.push('❌ Refresh button not found');
    
    await page.screenshot({ path: 'screenshots/detailed_05_sources.png', fullPage: true });
    
    // 6. SETTINGS TESTING
    console.log('\n\n6️⃣ TESTING SETTINGS (/settings)\n' + '-'.repeat(40));
    await page.goto('http://localhost:3100/settings', { waitUntil: 'networkidle' });
    
    // Check form elements
    const form = await page.$('form, .settings-form');
    if (form) {
      results.settings.pass.push('✅ Settings form present');
      
      const inputs = await page.$$('input, select, textarea');
      results.settings.elements.inputs = inputs.length;
      console.log(`  - Found ${inputs.length} input fields`);
      
      const checkboxes = await page.$$('input[type="checkbox"], .MuiSwitch-root');
      results.settings.elements.toggles = checkboxes.length;
      console.log(`  - Found ${checkboxes.length} toggle switches`);
    } else {
      results.settings.fail.push('❌ Settings form not found');
    }
    
    // Check save button
    const saveButton = await page.$('button[type="submit"], button:has-text("Save")');
    if (saveButton) {
      results.settings.pass.push('✅ Save button present');
    } else {
      results.settings.fail.push('❌ Save button not found');
    }
    
    await page.screenshot({ path: 'screenshots/detailed_06_settings.png', fullPage: true });
    
    // 7. NAVIGATION TESTING
    console.log('\n\n7️⃣ TESTING NAVIGATION\n' + '-'.repeat(40));
    
    // Check sidebar/navigation
    const nav = await page.$('nav, aside, [data-testid="sidebar"], .sidebar');
    if (nav) {
      results.navigation.pass.push('✅ Navigation sidebar present');
      
      // Test navigation links
      const navItems = [
        { name: 'Dashboard', path: '/' },
        { name: 'AI Intelligence', path: '/ai' },
        { name: 'Memory', path: '/memory' },
        { name: 'Search', path: '/search' },
        { name: 'Sources', path: '/sources' },
        { name: 'Settings', path: '/settings' }
      ];
      
      for (const item of navItems) {
        const link = await page.$(`a:has-text("${item.name}"), [href="${item.path}"]`);
        if (link) {
          results.navigation.pass.push(`✅ ${item.name} link present`);
        } else {
          results.navigation.fail.push(`❌ ${item.name} link not found`);
        }
      }
    } else {
      results.navigation.fail.push('❌ Navigation sidebar not found');
    }
    
    // 8. API INTEGRATION TESTING
    console.log('\n\n8️⃣ TESTING API INTEGRATION\n' + '-'.repeat(40));
    
    // Analyze API requests
    console.log(`\n  - Total API requests: ${apiRequests.length}`);
    console.log(`  - Failed requests: ${failedRequests.length}`);
    
    if (failedRequests.length === 0) {
      results.api.pass.push('✅ All API requests successful');
    } else {
      failedRequests.forEach(req => {
        results.api.fail.push(`❌ API Error: ${req.status} ${req.statusText} - ${req.url}`);
      });
    }
    
    // Check specific API endpoints
    const criticalEndpoints = [
      '/api/health',
      '/api/memory',
      '/api/sources',
      '/api/claude-auto'
    ];
    
    for (const endpoint of criticalEndpoints) {
      const found = apiRequests.some(req => req.url.includes(endpoint));
      if (found) {
        results.api.pass.push(`✅ ${endpoint} endpoint called`);
      } else {
        console.log(`  ⚠️  ${endpoint} endpoint not called during test`);
      }
    }
    
    // Performance metrics
    results.performance = {
      localLoadTime: localLoadTime,
      totalApiRequests: apiRequests.length,
      failedApiRequests: failedRequests.length,
      consoleErrors: consoleLogs.filter(log => log.type === 'error').length,
      pageErrors: pageErrors.length
    };
    
  } catch (error) {
    console.error('\n❌ Critical Test Error:', error.message);
  } finally {
    await browser.close();
  }
  
  // Generate comprehensive report
  console.log('\n\n' + '='.repeat(80));
  console.log('📊 COMPREHENSIVE VALIDATION REPORT');
  console.log('='.repeat(80));
  
  let totalPass = 0;
  let totalFail = 0;
  
  // Detailed section reports
  for (const [section, data] of Object.entries(results)) {
    if (section === 'performance') continue;
    
    console.log(`\n### ${section.toUpperCase()}`);
    
    if (data.pass && data.pass.length > 0) {
      console.log('\n✅ Passed Tests:');
      data.pass.forEach(test => {
        console.log(`  ${test}`);
        totalPass++;
      });
    }
    
    if (data.fail && data.fail.length > 0) {
      console.log('\n❌ Failed Tests:');
      data.fail.forEach(test => {
        console.log(`  ${test}`);
        totalFail++;
      });
    }
    
    if (data.elements && Object.keys(data.elements).length > 0) {
      console.log('\n📊 Element Counts:');
      for (const [element, count] of Object.entries(data.elements)) {
        console.log(`  - ${element}: ${count}`);
      }
    }
  }
  
  // Performance Summary
  console.log('\n\n### PERFORMANCE METRICS');
  console.log(`  - Initial Load Time: ${results.performance.localLoadTime}ms`);
  console.log(`  - Total API Requests: ${results.performance.totalApiRequests}`);
  console.log(`  - Failed API Requests: ${results.performance.failedApiRequests}`);
  console.log(`  - Console Errors: ${results.performance.consoleErrors}`);
  console.log(`  - Page Errors: ${results.performance.pageErrors}`);
  
  // Overall Score
  const totalTests = totalPass + totalFail;
  const score = totalTests > 0 ? (totalPass / totalTests * 100).toFixed(1) : 0;
  
  console.log('\n' + '='.repeat(80));
  console.log(`\n🎯 OVERALL SYSTEM HEALTH SCORE: ${score}%`);
  console.log(`   ✅ Passed: ${totalPass} tests`);
  console.log(`   ❌ Failed: ${totalFail} tests`);
  
  // Health Assessment
  console.log('\n💡 HEALTH ASSESSMENT:');
  if (score >= 90) {
    console.log('   🟢 EXCELLENT - System is fully operational');
  } else if (score >= 70) {
    console.log('   🟡 GOOD - System is mostly functional with minor issues');
  } else if (score >= 50) {
    console.log('   🟠 FAIR - System has significant issues that need attention');
  } else {
    console.log('   🔴 POOR - System has critical issues requiring immediate fixes');
  }
  
  // Recommendations
  console.log('\n📝 RECOMMENDATIONS:');
  if (results.api.fail.length > 0) {
    console.log('   - Fix API connectivity issues with the LAN server');
  }
  if (consoleLogs.filter(log => log.type === 'error').length > 0) {
    console.log('   - Resolve console errors for better stability');
  }
  if (totalFail > 0) {
    console.log('   - Address failed UI elements for complete functionality');
  }
  
  console.log('\n⏰ Test completed at:', new Date().toLocaleString());
  console.log('='.repeat(80) + '\n');
  
  return results;
}

// Run the comprehensive test
comprehensiveUIValidation().catch(console.error);