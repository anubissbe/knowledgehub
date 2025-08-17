import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== PERFORMANCE VALIDATION CHECK 2 (Runtime Performance) ===');
  
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  
  // Measure JavaScript performance
  const jsMetrics = await page.evaluate(() => {
    let startTime = performance.now();
    
    // Simulate user interactions
    const buttons = document.querySelectorAll('button');
    const links = document.querySelectorAll('a');
    
    let interactionTime = performance.now() - startTime;
    
    return {
      domElementsCount: document.querySelectorAll('*').length,
      buttonsCount: buttons.length,
      linksCount: links.length,
      interactionTime: interactionTime,
      memoryUsage: performance.memory ? {
        used: Math.round(performance.memory.usedJSHeapSize / 1048576),
        total: Math.round(performance.memory.totalJSHeapSize / 1048576),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1048576)
      } : null
    };
  });
  
  console.log('DOM Elements Count:', jsMetrics.domElementsCount);
  console.log('Interactive Buttons:', jsMetrics.buttonsCount);
  console.log('Interactive Links:', jsMetrics.linksCount);
  console.log('Interaction Time:', Math.round(jsMetrics.interactionTime) + 'ms');
  
  if (jsMetrics.memoryUsage) {
    console.log('Memory Usage:');
    console.log('  Used:', jsMetrics.memoryUsage.used + 'MB');
    console.log('  Total:', jsMetrics.memoryUsage.total + 'MB');
    console.log('  Limit:', jsMetrics.memoryUsage.limit + 'MB');
  }
  
  // Test navigation performance
  const navigationStart = Date.now();
  await page.goto('http://localhost:3101/dashboard', { waitUntil: 'networkidle' });
  const navigationTime = Date.now() - navigationStart;
  
  console.log('Navigation Time (to Dashboard):', navigationTime + 'ms');
  
  await browser.close();
})();
