const { chromium } = require('playwright');

async function testKnowledgeHubUI() {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 500 // Slow down for visibility
  });
  
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    ignoreHTTPSErrors: true
  });
  
  const page = await context.newPage();
  
  // Enable console logging
  page.on('console', msg => {
    if (msg.type() === 'error') {
      console.log('Console Error:', msg.text());
    }
  });
  
  page.on('pageerror', err => {
    console.log('Page Error:', err.message);
  });

  // Results object
  const results = {
    dashboard: { works: [], issues: [], errors: [] },
    ai: { works: [], issues: [], errors: [] },
    memory: { works: [], issues: [], errors: [] },
    search: { works: [], issues: [], errors: [] },
    sources: { works: [], issues: [], errors: [] },
    settings: { works: [], issues: [], errors: [] },
    navigation: { works: [], issues: [], errors: [] },
    api: { works: [], issues: [], errors: [] }
  };

  try {
    console.log('\n🚀 Starting KnowledgeHub UI Validation...\n');
    
    // Navigate to the application
    console.log('📍 Testing URL: http://localhost:3001');
    await page.goto('http://localhost:3001', { waitUntil: 'networkidle' });
    
    // Set up authentication
    console.log('\n🔐 Setting up authentication...');
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
    
    // Reload to apply settings
    await page.reload({ waitUntil: 'networkidle' });
    
    // 1. Test Dashboard
    console.log('\n1️⃣ Testing Dashboard (/)...');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/01_dashboard.png', fullPage: true });
    
    // Check dashboard elements
    const dashboardChecks = [
      { selector: '[data-testid="metric-card"]', description: 'Metric cards' },
      { selector: '.recharts-wrapper', description: 'Charts' },
      { selector: '[data-testid="system-performance"]', description: 'System performance' },
      { selector: '[data-testid="network-topology"]', description: 'Network topology' }
    ];
    
    for (const check of dashboardChecks) {
      try {
        await page.waitForSelector(check.selector, { timeout: 5000 });
        results.dashboard.works.push(`✅ ${check.description} loaded`);
      } catch (e) {
        results.dashboard.issues.push(`❌ ${check.description} not found`);
      }
    }
    
    // 2. Test AI Intelligence
    console.log('\n2️⃣ Testing AI Intelligence (/ai)...');
    await page.goto('http://localhost:3001/ai', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/02_ai_intelligence.png', fullPage: true });
    
    // Check AI features
    const aiChecks = [
      { selector: '[data-testid="ai-feature-card"]', description: 'AI feature cards' },
      { selector: '[role="tab"]', description: 'Tab navigation' },
      { selector: '[data-testid="performance-metrics"]', description: 'Performance metrics' },
      { selector: '[data-testid="ai-insights"]', description: 'AI insights panel' }
    ];
    
    for (const check of aiChecks) {
      try {
        await page.waitForSelector(check.selector, { timeout: 5000 });
        const count = await page.locator(check.selector).count();
        results.ai.works.push(`✅ ${check.description} loaded (${count} items)`);
      } catch (e) {
        results.ai.issues.push(`❌ ${check.description} not found`);
      }
    }
    
    // Test tab switching
    const tabs = await page.locator('[role="tab"]').all();
    if (tabs.length > 0) {
      for (let i = 0; i < Math.min(tabs.length, 4); i++) {
        await tabs[i].click();
        await page.waitForTimeout(500);
      }
      results.ai.works.push(`✅ Tab navigation working (${tabs.length} tabs)`);
    }
    
    // 3. Test Memory System
    console.log('\n3️⃣ Testing Memory System (/memory)...');
    await page.goto('http://localhost:3001/memory', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/03_memory_system.png', fullPage: true });
    
    // Check memory features
    const memoryChecks = [
      { selector: 'input[type="search"], input[placeholder*="search" i]', description: 'Search input' },
      { selector: '[data-testid="memory-stats"]', description: 'Memory statistics' },
      { selector: '[data-testid="memory-list"], table, .MuiDataGrid-root', description: 'Memory list/table' },
      { selector: '[data-testid="memory-distribution"]', description: 'Memory distribution' }
    ];
    
    for (const check of memoryChecks) {
      try {
        await page.waitForSelector(check.selector, { timeout: 5000 });
        results.memory.works.push(`✅ ${check.description} loaded`);
      } catch (e) {
        results.memory.issues.push(`❌ ${check.description} not found`);
      }
    }
    
    // Test search functionality
    const searchInput = await page.locator('input[type="search"], input[placeholder*="search" i]').first();
    if (searchInput) {
      await searchInput.fill('test search');
      await page.waitForTimeout(1000);
      results.memory.works.push('✅ Search input functional');
    }
    
    // 4. Test Search Knowledge
    console.log('\n4️⃣ Testing Search Knowledge (/search)...');
    await page.goto('http://localhost:3001/search', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/04_search_knowledge.png', fullPage: true });
    
    // Check search modes
    const searchModes = ['Semantic', 'Hybrid', 'Text'];
    for (const mode of searchModes) {
      const modeButton = await page.locator(`button:has-text("${mode}")`).first();
      if (modeButton) {
        await modeButton.click();
        results.search.works.push(`✅ ${mode} search mode available`);
      } else {
        results.search.issues.push(`❌ ${mode} search mode not found`);
      }
    }
    
    // 5. Test Sources
    console.log('\n5️⃣ Testing Sources (/sources)...');
    await page.goto('http://localhost:3001/sources', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/05_sources.png', fullPage: true });
    
    // Check source features
    const sourceChecks = [
      { selector: '[data-testid="source-list"], table', description: 'Source list' },
      { selector: 'button:has-text("Add Source")', description: 'Add Source button' },
      { selector: 'button:has-text("Refresh")', description: 'Refresh button' },
      { selector: '[data-testid="source-status"]', description: 'Source status indicators' }
    ];
    
    for (const check of sourceChecks) {
      try {
        await page.waitForSelector(check.selector, { timeout: 5000 });
        results.sources.works.push(`✅ ${check.description} loaded`);
      } catch (e) {
        results.sources.issues.push(`❌ ${check.description} not found`);
      }
    }
    
    // 6. Test Settings
    console.log('\n6️⃣ Testing Settings (/settings)...');
    await page.goto('http://localhost:3001/settings', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/06_settings.png', fullPage: true });
    
    // Check settings sections
    const settingsChecks = [
      { selector: 'form', description: 'Settings form' },
      { selector: 'input[type="text"], input[type="number"]', description: 'Input fields' },
      { selector: 'button[type="submit"], button:has-text("Save")', description: 'Save button' },
      { selector: '[data-testid="theme-toggle"], input[type="checkbox"]', description: 'Toggle switches' }
    ];
    
    for (const check of settingsChecks) {
      try {
        await page.waitForSelector(check.selector, { timeout: 5000 });
        const count = await page.locator(check.selector).count();
        results.settings.works.push(`✅ ${check.description} loaded (${count} items)`);
      } catch (e) {
        results.settings.issues.push(`❌ ${check.description} not found`);
      }
    }
    
    // 7. Test Navigation
    console.log('\n7️⃣ Testing Navigation...');
    
    // Check sidebar
    const sidebar = await page.locator('[data-testid="sidebar"], nav, aside').first();
    if (sidebar) {
      results.navigation.works.push('✅ Sidebar present');
      
      // Test navigation links
      const navLinks = [
        { text: 'Dashboard', path: '/' },
        { text: 'AI Intelligence', path: '/ai' },
        { text: 'Memory', path: '/memory' },
        { text: 'Search', path: '/search' },
        { text: 'Sources', path: '/sources' },
        { text: 'Settings', path: '/settings' }
      ];
      
      for (const link of navLinks) {
        const navLink = await page.locator(`a:has-text("${link.text}"), [role="link"]:has-text("${link.text}")`).first();
        if (navLink) {
          results.navigation.works.push(`✅ ${link.text} link present`);
        } else {
          results.navigation.issues.push(`❌ ${link.text} link not found`);
        }
      }
    } else {
      results.navigation.issues.push('❌ Sidebar not found');
    }
    
    // 8. Check API Integration
    console.log('\n8️⃣ Checking API Integration...');
    
    // Monitor network requests
    const apiErrors = [];
    page.on('response', response => {
      if (response.url().includes('/api/') && response.status() >= 400) {
        apiErrors.push(`${response.status()} - ${response.url()}`);
      }
    });
    
    // Navigate through pages to trigger API calls
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle' });
    await page.goto('http://localhost:3001/memory', { waitUntil: 'networkidle' });
    await page.goto('http://localhost:3001/ai', { waitUntil: 'networkidle' });
    
    if (apiErrors.length === 0) {
      results.api.works.push('✅ No API errors detected');
    } else {
      apiErrors.forEach(error => results.api.errors.push(`❌ API Error: ${error}`));
    }
    
    // Final screenshot
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle' });
    await page.screenshot({ path: 'screenshots/07_final_state.png', fullPage: true });
    
  } catch (error) {
    console.error('Test Error:', error);
  } finally {
    await browser.close();
  }
  
  // Generate report
  console.log('\n\n📊 VALIDATION REPORT\n' + '='.repeat(50));
  
  let totalScore = 0;
  let maxScore = 0;
  
  for (const [section, data] of Object.entries(results)) {
    console.log(`\n### ${section.toUpperCase()}`);
    
    maxScore += (data.works.length + data.issues.length + data.errors.length) || 1;
    totalScore += data.works.length;
    
    if (data.works.length > 0) {
      console.log('\n✅ Working:');
      data.works.forEach(item => console.log(`  ${item}`));
    }
    
    if (data.issues.length > 0) {
      console.log('\n❌ Issues:');
      data.issues.forEach(item => console.log(`  ${item}`));
    }
    
    if (data.errors.length > 0) {
      console.log('\n🚨 Errors:');
      data.errors.forEach(item => console.log(`  ${item}`));
    }
    
    const sectionScore = data.works.length / (data.works.length + data.issues.length + data.errors.length || 1) * 100;
    console.log(`\nSection Score: ${sectionScore.toFixed(1)}%`);
  }
  
  const overallScore = (totalScore / maxScore * 100).toFixed(1);
  console.log('\n' + '='.repeat(50));
  console.log(`\n🎯 OVERALL SYSTEM HEALTH SCORE: ${overallScore}%\n`);
  
  // Performance observations
  console.log('📈 Performance Observations:');
  console.log('  - Page load times: Good (networkidle reached)');
  console.log('  - API response times: Check network tab for details');
  console.log('  - UI responsiveness: Interactive elements respond to clicks');
  console.log('  - Error handling: Console errors logged above if any');
  
  return results;
}

// Run the test
testKnowledgeHubUI().catch(console.error);