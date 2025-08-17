import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  const errors = [];
  const warnings = [];
  const networkErrors = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
    if (msg.type() === 'warning') warnings.push(msg.text());
  });
  
  page.on('response', response => {
    if (!response.ok() && !response.url().includes('hot-update')) {
      networkErrors.push(response.status() + ' ' + response.url());
    }
  });
  
  console.log('=== CONSOLE VALIDATION CHECK 3 (Extended Usage) ===');
  
  // Test multiple routes
  const routes = [
    'http://localhost:3101/',
    'http://localhost:3101/dashboard', 
    'http://localhost:3101/ai',
    'http://localhost:3101/memory',
    'http://localhost:3101/search'
  ];
  
  for (const route of routes) {
    console.log(`Testing route: ${route}`);
    await page.goto(route, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
  }
  
  console.log('\n--- FINAL CONSOLE VALIDATION RESULTS ---');
  console.log('Console Errors:', errors.length);
  errors.forEach((err, i) => console.log(`ERROR ${i+1}:`, err));
  console.log('Console Warnings:', warnings.length);
  warnings.forEach((warn, i) => console.log(`WARN ${i+1}:`, warn));
  console.log('Network Errors:', networkErrors.length);
  networkErrors.forEach((err, i) => console.log(`NETWORK ${i+1}:`, err));
  
  await browser.close();
})();
