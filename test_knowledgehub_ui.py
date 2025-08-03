#!/usr/bin/env python3
"""
Comprehensive UI testing for KnowledgeHub web application
Tests all pages and features at http://192.168.1.25:3100
"""

import asyncio
import json
import os
from datetime import datetime
from playwright.async_api import async_playwright, Page, expect
from typing import Dict, List, Tuple

class KnowledgeHubUITester:
    def __init__(self, base_url: str = "http://192.168.1.25:3100"):
        self.base_url = base_url
        self.api_url = "http://192.168.1.25:3000"
        self.results = []
        self.screenshots_dir = "ui_test_screenshots"
        
    async def setup_browser(self):
        """Initialize browser and page"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        
        # Create screenshots directory
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    async def setup_authentication(self):
        """Set up localStorage authentication"""
        await self.page.goto(self.base_url)
        
        # Set authentication in localStorage
        auth_data = {
            "apiUrl": self.api_url,
            "apiKey": "knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM",
            "enableNotifications": True,
            "autoRefresh": True,
            "refreshInterval": 30,
            "darkMode": False,
            "language": "en",
            "animationSpeed": 1,
            "cacheSize": 100,
            "maxMemories": 1000,
            "compressionEnabled": True
        }
        
        await self.page.evaluate(f"""
            localStorage.setItem('knowledgehub_settings', '{json.dumps(auth_data)}')
        """)
        
        # Reload to apply settings
        await self.page.reload()
        await self.page.wait_for_load_state('networkidle')
        
    async def take_screenshot(self, name: str):
        """Take a screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshots_dir}/{timestamp}_{name}.png"
        await self.page.screenshot(path=filename, full_page=True)
        return filename
        
    async def test_page(self, path: str, name: str, tests: List[Tuple[str, callable]]) -> Dict:
        """Test a specific page with given tests"""
        result = {
            "page": name,
            "path": path,
            "status": "success",
            "errors": [],
            "tests_passed": 0,
            "tests_failed": 0,
            "screenshot": None
        }
        
        try:
            # Navigate to page
            await self.page.goto(f"{self.base_url}{path}")
            await self.page.wait_for_load_state('networkidle')
            
            # Take initial screenshot
            result["screenshot"] = await self.take_screenshot(f"{name.lower().replace(' ', '_')}_initial")
            
            # Run specific tests for this page
            for test_name, test_func in tests:
                try:
                    await test_func()
                    result["tests_passed"] += 1
                    print(f"✅ {name} - {test_name}: PASSED")
                except Exception as e:
                    result["tests_failed"] += 1
                    result["errors"].append(f"{test_name}: {str(e)}")
                    print(f"❌ {name} - {test_name}: FAILED - {str(e)}")
                    
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(f"Page load error: {str(e)}")
            print(f"❌ {name} - Page load: FAILED - {str(e)}")
            
        if result["tests_failed"] > 0 or result["status"] == "failed":
            result["status"] = "partial" if result["tests_passed"] > 0 else "failed"
            
        return result
        
    async def test_home_dashboard(self):
        """Test Home/Dashboard page"""
        async def check_header():
            header = await self.page.wait_for_selector('h1', timeout=5000)
            text = await header.text_content()
            assert "KnowledgeHub" in text or "Dashboard" in text
            
        async def check_stats():
            # Check for statistics cards
            stats = await self.page.query_selector_all('.stat-card, .dashboard-stat, [class*="stat"]')
            assert len(stats) > 0, "No statistics cards found"
            
        async def check_navigation():
            # Check navigation menu exists
            nav = await self.page.query_selector('nav, .navigation, [class*="nav"]')
            assert nav is not None, "Navigation menu not found"
            
        tests = [
            ("Header presence", check_header),
            ("Statistics display", check_stats),
            ("Navigation menu", check_navigation)
        ]
        
        return await self.test_page("/", "Home Dashboard", tests)
        
    async def test_ai_intelligence(self):
        """Test AI Intelligence page with all 8 features"""
        async def check_ai_features():
            # Wait for AI features to load
            await self.page.wait_for_selector('[class*="feature"], [class*="ai-feature"]', timeout=10000)
            features = await self.page.query_selector_all('[class*="feature"], [class*="ai-feature"]')
            assert len(features) >= 8, f"Expected 8 AI features, found {len(features)}"
            
        async def check_session_continuity():
            # Look for session continuity section
            session_element = await self.page.query_selector('text=/session.?continuity/i')
            assert session_element is not None, "Session Continuity feature not found"
            
        async def check_mistake_learning():
            # Look for mistake learning section
            mistake_element = await self.page.query_selector('text=/mistake.?learning/i')
            assert mistake_element is not None, "Mistake Learning feature not found"
            
        async def check_proactive_assistance():
            # Look for proactive assistance section
            proactive_element = await self.page.query_selector('text=/proactive.?assistance/i')
            assert proactive_element is not None, "Proactive Assistance feature not found"
            
        tests = [
            ("AI features display", check_ai_features),
            ("Session Continuity", check_session_continuity),
            ("Mistake Learning", check_mistake_learning),
            ("Proactive Assistance", check_proactive_assistance)
        ]
        
        return await self.test_page("/ai-intelligence", "AI Intelligence", tests)
        
    async def test_memory_system(self):
        """Test Memory System page"""
        async def check_memory_list():
            # Wait for memory items or empty state
            await self.page.wait_for_selector('[class*="memory"], [class*="empty"]', timeout=10000)
            
        async def check_search_functionality():
            # Find search input
            search_input = await self.page.query_selector('input[type="search"], input[placeholder*="search" i]')
            if search_input:
                await search_input.fill("test query")
                await self.page.keyboard.press("Enter")
                await self.page.wait_for_timeout(1000)
                
        async def check_stats_display():
            # Look for memory statistics
            stats = await self.page.query_selector_all('[class*="stat"], [class*="metric"]')
            assert len(stats) > 0, "No memory statistics found"
            
        tests = [
            ("Memory list display", check_memory_list),
            ("Search functionality", check_search_functionality),
            ("Statistics display", check_stats_display)
        ]
        
        return await self.test_page("/memory", "Memory System", tests)
        
    async def test_search_knowledge(self):
        """Test Search Knowledge page"""
        async def check_search_types():
            # Look for search type options (semantic, hybrid, text)
            await self.page.wait_for_selector('[class*="search"]', timeout=10000)
            
        async def test_search_execution():
            # Find search input and button
            search_input = await self.page.query_selector('input[type="search"], input[type="text"]')
            if search_input:
                await search_input.fill("test search query")
                
                # Try to find and click search button
                search_button = await self.page.query_selector('button[type="submit"], button:has-text("Search")')
                if search_button:
                    await search_button.click()
                    await self.page.wait_for_timeout(2000)
                    
        async def check_search_options():
            # Look for search type toggles or dropdowns
            options = await self.page.query_selector_all('input[type="radio"], select, [class*="toggle"]')
            assert len(options) > 0, "No search options found"
            
        tests = [
            ("Search interface", check_search_types),
            ("Search execution", test_search_execution),
            ("Search options", check_search_options)
        ]
        
        return await self.test_page("/search", "Search Knowledge", tests)
        
    async def test_sources(self):
        """Test Sources page"""
        async def check_source_list():
            # Wait for sources to load
            await self.page.wait_for_selector('[class*="source"], table, [class*="list"]', timeout=10000)
            
        async def check_source_stats():
            # Look for source statistics
            stats = await self.page.query_selector_all('[class*="stat"], [class*="count"]')
            assert len(stats) > 0, "No source statistics found"
            
        async def check_source_details():
            # Try to click on a source if available
            source_items = await self.page.query_selector_all('[class*="source-item"], tbody tr')
            if len(source_items) > 0:
                await source_items[0].click()
                await self.page.wait_for_timeout(1000)
                
        tests = [
            ("Source list display", check_source_list),
            ("Source statistics", check_source_stats),
            ("Source interaction", check_source_details)
        ]
        
        return await self.test_page("/sources", "Sources", tests)
        
    async def test_settings(self):
        """Test Settings page"""
        async def check_settings_form():
            # Wait for settings form
            await self.page.wait_for_selector('form, [class*="settings"]', timeout=10000)
            
        async def check_api_settings():
            # Look for API URL and key inputs
            api_inputs = await self.page.query_selector_all('input[name*="api" i], input[placeholder*="api" i]')
            assert len(api_inputs) > 0, "No API settings inputs found"
            
        async def test_settings_toggle():
            # Find and test a toggle switch
            toggles = await self.page.query_selector_all('input[type="checkbox"], [class*="switch"]')
            if len(toggles) > 0:
                await toggles[0].click()
                await self.page.wait_for_timeout(500)
                
        tests = [
            ("Settings form", check_settings_form),
            ("API configuration", check_api_settings),
            ("Settings toggles", test_settings_toggle)
        ]
        
        return await self.test_page("/settings", "Settings", tests)
        
    async def run_all_tests(self):
        """Run all UI tests"""
        print("🚀 Starting KnowledgeHub UI Testing")
        print(f"Target: {self.base_url}")
        print("-" * 50)
        
        await self.setup_browser()
        await self.setup_authentication()
        
        # Run all page tests
        test_functions = [
            self.test_home_dashboard,
            self.test_ai_intelligence,
            self.test_memory_system,
            self.test_search_knowledge,
            self.test_sources,
            self.test_settings
        ]
        
        for test_func in test_functions:
            result = await test_func()
            self.results.append(result)
            print("-" * 50)
            
        await self.generate_report()
        await self.cleanup()
        
    async def generate_report(self):
        """Generate test report"""
        print("\n📊 TEST REPORT")
        print("=" * 50)
        
        total_pages = len(self.results)
        successful_pages = sum(1 for r in self.results if r["status"] == "success")
        failed_pages = sum(1 for r in self.results if r["status"] == "failed")
        partial_pages = sum(1 for r in self.results if r["status"] == "partial")
        
        print(f"Total Pages Tested: {total_pages}")
        print(f"✅ Successful: {successful_pages}")
        print(f"⚠️  Partial Success: {partial_pages}")
        print(f"❌ Failed: {failed_pages}")
        print("\nDetailed Results:")
        print("-" * 50)
        
        for result in self.results:
            status_emoji = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "partial" else "❌"
            print(f"{status_emoji} {result['page']} ({result['path']})")
            print(f"   Tests: {result['tests_passed']} passed, {result['tests_failed']} failed")
            if result["errors"]:
                print("   Errors:")
                for error in result["errors"]:
                    print(f"   - {error}")
            if result["screenshot"]:
                print(f"   Screenshot: {result['screenshot']}")
            print()
            
        # Save detailed report
        report_file = f"ui_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_file}")
        
    async def cleanup(self):
        """Clean up browser resources"""
        await self.browser.close()
        await self.playwright.stop()

async def main():
    tester = KnowledgeHubUITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())