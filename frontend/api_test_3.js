import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== API INTEGRATION CHECK 3 (Loading Behavior) ===');
  
  // Slow down network to see loading states
  const client = await page.context().newCDPSession(page);
  await client.send('Network.enable');
  await client.send('Network.emulateNetworkConditions', {
    offline: false,
    downloadThroughput: 100000, // 100kb/s
    uploadThroughput: 100000,
    latency: 100 // 100ms latency
  });
  
  const loadingStates = [];
  
  // Monitor for loading indicators during navigation
  page.on('response', response => {
    if (response.url().includes('/api/')) {
      loadingStates.push({
        url: response.url(),
        status: response.status(),
        timing: Date.now()
      });
    }
  });
  
  console.log('Testing with throttled network...');
  
  const startTime = Date.now();
  await page.goto('http://localhost:3101/dashboard', { waitUntil: 'networkidle', timeout: 30000 });
  const loadTime = Date.now() - startTime;
  
  console.log('Dashboard load time (throttled):', loadTime + 'ms');
  console.log('API responses during load:', loadingStates.length);
  
  // Check for loading indicators at different moments
  const loadingCheck = await page.evaluate(() => {
    const hasSpinners = document.querySelectorAll('.spinner, [data-loading="true"]').length;
    const hasSkeleton = document.querySelectorAll('.skeleton, .placeholder').length;
    const hasProgressBars = document.querySelectorAll('.progress, .progress-bar').length;
    
    return {
      spinners: hasSpinners,
      skeleton: hasSkeleton,
      progressBars: hasProgressBars
    };
  });
  
  console.log('Loading UI Elements:');
  console.log('  Spinners:', loadingCheck.spinners);
  console.log('  Skeleton screens:', loadingCheck.skeleton);
  console.log('  Progress bars:', loadingCheck.progressBars);
  
  // Reset network conditions
  await client.send('Network.emulateNetworkConditions', {
    offline: false,
    downloadThroughput: -1,
    uploadThroughput: -1,
    latency: 0
  });
  
  await browser.close();
})();
