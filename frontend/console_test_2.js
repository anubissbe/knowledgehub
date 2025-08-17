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
    if (!response.ok()) {
      networkErrors.push(response.status() + ' ' + response.url());
    }
  });
  
  console.log('=== CONSOLE VALIDATION CHECK 2 (Development Server) ===');
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  
  console.log('Console Errors:', errors.length);
  errors.slice(0, 5).forEach(err => console.log('ERROR:', err));
  console.log('Console Warnings:', warnings.length); 
  warnings.slice(0, 5).forEach(warn => console.log('WARN:', warn));
  console.log('Network Errors:', networkErrors.length);
  networkErrors.slice(0, 5).forEach(err => console.log('NETWORK:', err));
  
  await browser.close();
})();
