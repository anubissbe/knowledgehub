#!/usr/bin/env python3
"""
Headless UI testing for KnowledgeHub AI Intelligence page
"""

import asyncio
import json
import os
from datetime import datetime
from playwright.async_api import async_playwright, Page
from typing import Dict, List

class KnowledgeHubAITester:
    def __init__(self, base_url: str = "http://localhost:3100"):
        self.base_url = base_url
        self.api_url = "http://localhost:3000"
        self.screenshots_dir = f"ai_test_screenshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        
    async def setup(self):
        """Initialize browser in headless mode"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,  # Run headless
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        self.page = await self.context.new_page()
        
        # Enable console logging
        self.page.on("console", lambda msg: print(f"CONSOLE {msg.type}: {msg.text}"))
        self.page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))
        
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    async def set_authentication(self):
        """Set authentication in localStorage"""
        print("Setting authentication...")
        await self.page.goto(self.base_url)
        
        # Set localStorage
        await self.page.evaluate("""
            localStorage.setItem('knowledgehub_settings', JSON.stringify({
                apiUrl: "http://192.168.1.25:3000",
                apiKey: "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM",
                enableNotifications: true,
                autoRefresh: true,
                refreshInterval: 30,
                darkMode: false,
                language: "en",
                animationSpeed: 1,
                cacheSize: 100,
                maxMemories: 1000,
                compressionEnabled: true
            }));
        """)
        
        # Reload to apply settings
        await self.page.reload()
        await self.page.wait_for_timeout(2000)
        
    async def wait_for_react_app(self):
        """Wait for React app to load"""
        print("Waiting for React app to initialize...")
        
        try:
            await self.page.wait_for_function(
                "document.querySelector('#root') && document.querySelector('#root').children.length > 0",
                timeout=30000
            )
            await self.page.wait_for_timeout(2000)
            print("✓ React app loaded")
            return True
        except Exception as e:
            print(f"✗ React app failed to load: {e}")
            return False
            
    async def test_ai_intelligence_page(self):
        """Test the AI Intelligence page specifically"""
        print("\n=== Testing AI Intelligence Page ===")
        
        # Navigate to AI Intelligence page
        print("1. Navigating to AI Intelligence page...")
        await self.page.goto(f"{self.base_url}/ai")
        await self.page.wait_for_timeout(3000)
        
        # Take initial screenshot
        await self.page.screenshot(path=f"{self.screenshots_dir}/01_ai_page_initial.png")
        
        # Check if AI feature cards are visible
        print("2. Checking for AI feature cards...")
        try:
            # Wait for feature cards container
            await self.page.wait_for_selector('.grid.grid-cols-1', timeout=10000)
            
            # Count feature cards
            feature_cards = await self.page.locator('.bg-white.rounded-lg.shadow-md').count()
            print(f"   Found {feature_cards} feature cards")
            self.results.append({"check": "Feature cards visible", "status": "PASS", "count": feature_cards})
            
            # Get all feature card titles
            titles = await self.page.locator('.text-lg.font-semibold').all_text_contents()
            print("   Feature cards found:")
            for title in titles:
                print(f"   - {title}")
                
        except Exception as e:
            print(f"   ✗ Failed to find feature cards: {e}")
            self.results.append({"check": "Feature cards visible", "status": "FAIL", "error": str(e)})
            
        # Test tab navigation
        print("\n3. Testing tab navigation...")
        tabs = ['All Features', 'Learning & Adaptation', 'Automation', 'Intelligence']
        
        for i, tab_name in enumerate(tabs):
            try:
                print(f"   Clicking on '{tab_name}' tab...")
                # Find and click tab
                tab_button = self.page.locator('button', has_text=tab_name)
                await tab_button.click()
                await self.page.wait_for_timeout(1000)
                
                # Take screenshot
                await self.page.screenshot(path=f"{self.screenshots_dir}/02_tab_{i+1}_{tab_name.replace(' ', '_').replace('&', 'and')}.png")
                
                # Check if tab is active
                is_active = await tab_button.evaluate("el => el.classList.contains('bg-blue-500') || el.classList.contains('text-blue-600')")
                print(f"   ✓ '{tab_name}' tab clicked - Active: {is_active}")
                self.results.append({"check": f"Tab '{tab_name}'", "status": "PASS" if is_active else "PARTIAL"})
                
            except Exception as e:
                print(f"   ✗ Failed to click '{tab_name}' tab: {e}")
                self.results.append({"check": f"Tab '{tab_name}'", "status": "FAIL", "error": str(e)})
                
        # Test clicking on a feature card
        print("\n4. Testing feature card interaction...")
        try:
            # Click on first feature card
            first_card = self.page.locator('.bg-white.rounded-lg.shadow-md').first
            card_title = await first_card.locator('.text-lg.font-semibold').text_content()
            print(f"   Clicking on '{card_title}' card...")
            
            await first_card.click()
            await self.page.wait_for_timeout(1000)
            
            # Check if clicked (card might have hover effect or expanded state)
            await self.page.screenshot(path=f"{self.screenshots_dir}/03_feature_card_clicked.png")
            print(f"   ✓ Clicked on feature card")
            self.results.append({"check": "Feature card clickable", "status": "PASS"})
            
        except Exception as e:
            print(f"   ✗ Failed to click feature card: {e}")
            self.results.append({"check": "Feature card clickable", "status": "FAIL", "error": str(e)})
            
        # Check for performance chart
        print("\n5. Checking for performance chart...")
        try:
            # Look for chart container or canvas
            chart_found = await self.page.locator('canvas').count() > 0 or \
                         await self.page.locator('[class*="chart"]').count() > 0 or \
                         await self.page.locator('[class*="recharts"]').count() > 0
                         
            if chart_found:
                print("   ✓ Performance chart found")
                self.results.append({"check": "Performance chart", "status": "PASS"})
            else:
                print("   ✗ Performance chart not found")
                self.results.append({"check": "Performance chart", "status": "FAIL"})
                
        except Exception as e:
            print(f"   ✗ Error checking for chart: {e}")
            self.results.append({"check": "Performance chart", "status": "FAIL", "error": str(e)})
            
        # Check for AI Insights panel
        print("\n6. Checking for AI Insights panel...")
        try:
            # Look for insights section
            insights_found = await self.page.locator('text=/AI Insights|Insights|Recent Activity/i').count() > 0
            
            if insights_found:
                print("   ✓ AI Insights panel found")
                self.results.append({"check": "AI Insights panel", "status": "PASS"})
                
                # Try to get insights content
                insights_content = await self.page.locator('[class*="insight"], [class*="activity"]').all_text_contents()
                if insights_content:
                    print(f"   Found {len(insights_content)} insights/activities")
                    for content in insights_content[:3]:  # Show first 3
                        print(f"   - {content[:100]}...")
            else:
                print("   ✗ AI Insights panel not found")
                self.results.append({"check": "AI Insights panel", "status": "FAIL"})
                
        except Exception as e:
            print(f"   ✗ Error checking insights: {e}")
            self.results.append({"check": "AI Insights panel", "status": "FAIL", "error": str(e)})
            
        # Take final full page screenshot
        await self.page.screenshot(path=f"{self.screenshots_dir}/04_ai_page_final_full.png", full_page=True)
        
    async def cleanup(self):
        """Clean up browser resources"""
        await self.browser.close()
        await self.playwright.stop()
        
    async def run_tests(self):
        """Run all tests"""
        try:
            await self.setup()
            await self.set_authentication()
            
            if await self.wait_for_react_app():
                await self.test_ai_intelligence_page()
            else:
                print("Failed to load React app, aborting tests")
                
        finally:
            await self.cleanup()
            
        # Generate report
        self.generate_report()
        
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("AI INTELLIGENCE PAGE TEST REPORT")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"URL: {self.base_url}/ai")
        print(f"Screenshots: {self.screenshots_dir}/")
        print("\nTest Results:")
        print("-"*60)
        
        passed = 0
        failed = 0
        partial = 0
        
        for result in self.results:
            status = result['status']
            check = result['check']
            
            if status == 'PASS':
                passed += 1
                print(f"✓ {check}: PASSED")
            elif status == 'PARTIAL':
                partial += 1
                print(f"⚠ {check}: PARTIAL")
            else:
                failed += 1
                error = result.get('error', '')
                print(f"✗ {check}: FAILED{' - ' + error if error else ''}")
                
        print("-"*60)
        print(f"Total: {len(self.results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Partial: {partial}")
        print(f"Success Rate: {(passed / len(self.results) * 100):.1f}%")
        
        # Save report to JSON
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "url": f"{self.base_url}/ai",
            "screenshots_dir": self.screenshots_dir,
            "results": self.results,
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "partial": partial,
                "success_rate": f"{(passed / len(self.results) * 100):.1f}%"
            }
        }
        
        report_file = f"ai_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\nDetailed report saved to: {report_file}")
        
        # Overall assessment
        print("\n" + "="*60)
        if failed == 0:
            print("✅ AI INTELLIGENCE PAGE IS FULLY FUNCTIONAL!")
        elif passed > failed:
            print("⚠️  AI Intelligence page is partially functional")
            print("   Some features may need attention")
        else:
            print("❌ AI Intelligence page has significant issues")
            print("   Most features are not working properly")
            
async def main():
    tester = KnowledgeHubAITester()
    await tester.run_tests()

if __name__ == "__main__":
    print("🚀 Starting KnowledgeHub AI Intelligence Page Test")
    print("Target: http://192.168.1.25:3100/ai")
    print("-"*60)
    asyncio.run(main())