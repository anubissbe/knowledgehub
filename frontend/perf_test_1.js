import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== PERFORMANCE VALIDATION CHECK 1 (Load Times) ===');
  
  const routes = [
    { path: '/', name: 'Home' },
    { path: '/dashboard', name: 'Dashboard' },
    { path: '/ai', name: 'AI Intelligence' },
    { path: '/memory', name: 'Memory System' },
    { path: '/search', name: 'Search' }
  ];
  
  for (const route of routes) {
    console.log('Testing:', route.name);
    
    const start = Date.now();
    await page.goto('http://localhost:3101' + route.path, { waitUntil: 'networkidle' });
    const loadTime = Date.now() - start;
    
    // Get performance metrics
    const performanceMetrics = await page.evaluate(() => {
      const timing = performance.timing;
      return {
        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
        loadComplete: timing.loadEventEnd - timing.navigationStart,
        firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0,
        firstContentfulPaint: performance.getEntriesByType('paint')[1]?.startTime || 0
      };
    });
    
    console.log('  Total Load Time:', loadTime + 'ms');
    console.log('  DOM Content Loaded:', performanceMetrics.domContentLoaded + 'ms');
    console.log('  Load Complete:', performanceMetrics.loadComplete + 'ms');
    console.log('  First Paint:', Math.round(performanceMetrics.firstPaint) + 'ms');
    console.log('  First Contentful Paint:', Math.round(performanceMetrics.firstContentfulPaint) + 'ms');
    console.log('');
  }
  
  await browser.close();
})();
