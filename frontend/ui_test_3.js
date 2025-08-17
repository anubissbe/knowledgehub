import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== UI/UX VALIDATION CHECK 3 (Responsive Design) ===');
  
  const viewports = [
    { name: 'Mobile Portrait', width: 375, height: 667 },
    { name: 'Mobile Landscape', width: 667, height: 375 },
    { name: 'Tablet Portrait', width: 768, height: 1024 },
    { name: 'Desktop', width: 1920, height: 1080 }
  ];
  
  for (const viewport of viewports) {
    console.log('Testing viewport:', viewport.name);
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    
    await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);
    
    // Check if content fits in viewport
    const bodyScrollWidth = await page.evaluate(() => document.body.scrollWidth);
    const bodyScrollHeight = await page.evaluate(() => document.body.scrollHeight);
    
    console.log('  Content width:', bodyScrollWidth, 'vs viewport:', viewport.width);
    console.log('  Content height:', bodyScrollHeight, 'vs viewport:', viewport.height);
    console.log('  Horizontal overflow:', bodyScrollWidth > viewport.width);
    
    // Take screenshot for this viewport
    await page.screenshot({ 
      path: 'ui_responsive_' + viewport.name.replace(' ', '_') + '.png',
      fullPage: false 
    });
  }
  
  await browser.close();
})();
