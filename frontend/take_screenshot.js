const { chromium } = require('playwright')

async function takeScreenshot() {
  console.log('📸 Taking Phase 5 UI screenshots...')
  
  const browser = await chromium.launch({ 
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage']
  })
  
  const page = await browser.newPage({
    viewport: { width: 1920, height: 1080 }
  })

  try {
    // Take dashboard screenshot
    await page.goto('http://192.168.1.25:3100', { waitUntil: 'networkidle' })
    await page.waitForTimeout(3000) // Wait for animations and data loading
    
    await page.screenshot({ 
      path: 'phase5-dashboard-proof.png',
      fullPage: true 
    })
    
    console.log('✅ Dashboard screenshot saved: phase5-dashboard-proof.png')
    
    // Try AI page if available
    try {
      await page.goto('http://192.168.1.25:3100/ai', { waitUntil: 'networkidle' })
      await page.waitForTimeout(2000)
      
      await page.screenshot({ 
        path: 'phase5-ai-page-proof.png',
        fullPage: true 
      })
      
      console.log('✅ AI page screenshot saved: phase5-ai-page-proof.png')
    } catch (err) {
      console.log('ℹ️  AI page not accessible, skipping screenshot')
    }
    
  } catch (error) {
    console.error('❌ Screenshot failed:', error.message)
  } finally {
    await browser.close()
  }
}

takeScreenshot()
EOF < /dev/null
