import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== PERFORMANCE VALIDATION CHECK 3 (Resource Optimization) ===');
  
  // Track network requests
  let resourceCount = { js: 0, css: 0, img: 0, other: 0 };
  let totalSize = 0;
  
  page.on('response', response => {
    const url = response.url();
    const contentType = response.headers()['content-type'] || '';
    
    if (url.includes('.js')) resourceCount.js++;
    else if (url.includes('.css')) resourceCount.css++;
    else if (contentType.includes('image')) resourceCount.img++;
    else resourceCount.other++;
    
    // Estimate size (not exact, but gives indication)
    totalSize += parseInt(response.headers()['content-length'] || '0', 10);
  });
  
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);
  
  console.log('Resource Counts:');
  console.log('  JavaScript files:', resourceCount.js);
  console.log('  CSS files:', resourceCount.css);
  console.log('  Images:', resourceCount.img);
  console.log('  Other resources:', resourceCount.other);
  console.log('  Total estimated size:', Math.round(totalSize / 1024) + 'KB');
  
  // Check for accessibility
  const a11yIssues = await page.evaluate(() => {
    const issues = [];
    
    // Check for alt attributes on images
    const images = document.querySelectorAll('img:not([alt])');
    if (images.length > 0) issues.push('Images without alt text: ' + images.length);
    
    // Check for form labels
    const inputs = document.querySelectorAll('input:not([aria-label]):not([aria-labelledby])');
    const unlabeledInputs = Array.from(inputs).filter(input => {
      const label = document.querySelector('label[for="' + input.id + '"]');
      return !label;
    });
    if (unlabeledInputs.length > 0) issues.push('Unlabeled inputs: ' + unlabeledInputs.length);
    
    // Check for heading hierarchy
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    if (headings.length === 0) issues.push('No heading elements found');
    
    return issues;
  });
  
  console.log('Accessibility Issues:');
  if (a11yIssues.length === 0) {
    console.log('  No major accessibility issues found');
  } else {
    a11yIssues.forEach(issue => console.log('  ' + issue));
  }
  
  await browser.close();
})();
