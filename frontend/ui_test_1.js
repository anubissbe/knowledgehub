import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== UI/UX VALIDATION CHECK 1 (Visual Consistency) ===');
  
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  
  // Check basic layout elements
  const navigation = await page.$('nav').catch(() => null);
  const header = await page.$('header').catch(() => null);
  const main = await page.$('main').catch(() => null);
  
  console.log('Navigation element present:', !!navigation);
  console.log('Header element present:', !!header);
  console.log('Main content present:', !!main);
  
  // Check for basic interactive elements
  const buttons = await page.$$('button');
  const links = await page.$$('a');
  const inputs = await page.$$('input');
  
  console.log('Buttons found:', buttons.length);
  console.log('Links found:', links.length); 
  console.log('Input elements found:', inputs.length);
  
  // Test navigation functionality
  const navLinks = await page.$$('nav a, [role="navigation"] a');
  console.log('Navigation links found:', navLinks.length);
  
  // Take screenshot for visual verification
  await page.screenshot({ path: 'ui_visual_check.png', fullPage: true });
  console.log('Screenshot saved: ui_visual_check.png');
  
  await browser.close();
})();
