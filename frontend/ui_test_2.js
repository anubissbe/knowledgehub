import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== UI/UX VALIDATION CHECK 2 (Interactive Functionality) ===');
  
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  
  // Test routing by direct navigation
  const routes = ['/dashboard', '/ai', '/memory', '/search'];
  
  for (const route of routes) {
    console.log('Testing route: ' + route);
    await page.goto('http://localhost:3101' + route, { waitUntil: 'networkidle' });
    
    const title = await page.title();
    const url = page.url();
    console.log('  Title: ' + title);
    console.log('  URL loaded correctly: ' + (url.indexOf(route) !== -1));
    
    await page.waitForTimeout(1000);
  }
  
  // Test common UI interactions
  await page.goto('http://localhost:3101/dashboard', { waitUntil: 'networkidle' });
  
  // Look for clickable elements
  const buttons = await page.$$('button');
  const links = await page.$$('a');
  console.log('Interactive buttons found:', buttons.length);
  console.log('Interactive links found:', links.length);
  
  await browser.close();
})();
