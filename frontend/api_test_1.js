import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== API INTEGRATION CHECK 1 (Network Requests) ===');
  
  const apiRequests = [];
  const failedRequests = [];
  
  page.on('request', request => {
    if (request.url().includes('/api/')) {
      apiRequests.push({
        url: request.url(),
        method: request.method()
      });
    }
  });
  
  page.on('response', response => {
    if (response.url().includes('/api/') && !response.ok()) {
      failedRequests.push({
        url: response.url(),
        status: response.status(),
        statusText: response.statusText()
      });
    }
  });
  
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  
  console.log('API Requests Made:', apiRequests.length);
  apiRequests.forEach((req, i) => {
    console.log(`  ${i+1}: ${req.method} ${req.url}`);
  });
  
  console.log('Failed API Requests:', failedRequests.length);
  failedRequests.forEach((req, i) => {
    console.log(`  ${i+1}: ${req.status} ${req.statusText} - ${req.url}`);
  });
  
  // Test different routes for API calls
  const routes = ['/dashboard', '/ai', '/memory'];
  for (const route of routes) {
    console.log('Testing API calls for route:', route);
    const routeApiRequests = [];
    
    page.on('request', request => {
      if (request.url().includes('/api/')) {
        routeApiRequests.push(request.url());
      }
    });
    
    await page.goto('http://localhost:3101' + route, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    
    console.log(`  API calls for ${route}:`, routeApiRequests.length);
  }
  
  await browser.close();
})();
