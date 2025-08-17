import puppeteer from 'playwright';

(async () => {
  const browser = await puppeteer.chromium.launch();
  const page = await browser.newPage();
  
  console.log('=== API INTEGRATION CHECK 2 (Error Handling) ===');
  
  await page.goto('http://localhost:3101', { waitUntil: 'networkidle' });
  
  // Check for error boundaries and loading states
  const errorElements = await page.evaluate(() => {
    const errors = document.querySelectorAll('[data-testid*="error"], .error, .alert-error');
    const loading = document.querySelectorAll('[data-testid*="loading"], .loading, .spinner');
    const empty = document.querySelectorAll('[data-testid*="empty"], .empty-state, .no-data');
    
    return {
      errorStates: errors.length,
      loadingStates: loading.length,
      emptyStates: empty.length,
      hasErrorBoundary: !!document.querySelector('[data-error-boundary]')
    };
  });
  
  console.log('Error State Elements:', errorElements.errorStates);
  console.log('Loading State Elements:', errorElements.loadingStates);
  console.log('Empty State Elements:', errorElements.emptyStates);
  console.log('Has Error Boundary:', errorElements.hasErrorBoundary);
  
  // Check different routes for error handling
  const routes = ['/dashboard', '/ai', '/memory', '/search'];
  for (const route of routes) {
    await page.goto('http://localhost:3101' + route, { waitUntil: 'networkidle' });
    
    const routeErrors = await page.evaluate(() => {
      const errorText = document.body.innerText.toLowerCase();
      return {
        hasErrorMessage: errorText.includes('error') || errorText.includes('failed'),
        hasLoadingIndicator: errorText.includes('loading') || errorText.includes('fetching'),
        hasDataPlaceholder: errorText.includes('no data') || errorText.includes('empty')
      };
    });
    
    console.log(`Route ${route}:`);
    console.log(`  Error messages visible: ${routeErrors.hasErrorMessage}`);
    console.log(`  Loading indicators: ${routeErrors.hasLoadingIndicator}`);
    console.log(`  Data placeholders: ${routeErrors.hasDataPlaceholder}`);
  }
  
  await browser.close();
})();
