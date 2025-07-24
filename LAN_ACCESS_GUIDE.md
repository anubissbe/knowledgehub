# KnowledgeHub LAN Access Guide

## 🌐 Access URLs
- **Web UI**: http://192.168.1.25:3100
- **API**: http://192.168.1.25:3000
- **AI Intelligence Dashboard**: http://192.168.1.25:3100/ai

## 🔧 Setup Instructions

### 1. Access the Web UI
Open your browser and navigate to: **http://192.168.1.25:3100**

### 2. Configure API Access
Open the browser console (F12) and run this command:

```javascript
localStorage.setItem('knowledgehub_settings', '{"apiUrl":"http://192.168.1.25:3000","apiKey":"knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM","enableNotifications":true,"autoRefresh":true,"refreshInterval":30,"darkMode":false,"language":"en","animationSpeed":1,"cacheSize":100,"maxMemories":1000,"compressionEnabled":true}')
```

### 3. Refresh the Page
Press F5 or click the refresh button

### 4. Navigate to AI Intelligence
Click on "AI Intelligence" in the sidebar menu

## 🎯 What You'll See

### AI Intelligence Dashboard
The AI Intelligence page displays 8 advanced features:

1. **Session Continuity** 🔄
   - Context preservation across sessions
   - Shows total sessions and memories

2. **Mistake Learning** 🎓
   - Learn from errors to prevent repetition
   - Displays patterns and lessons learned

3. **Proactive Assistant** 🧠
   - Anticipate needs and suggest actions
   - Shows prediction status

4. **Decision Reasoning** 🌳
   - Track and explain technical decisions
   - Lists decisions by category

5. **Code Evolution** 💻
   - Track code changes and patterns
   - Shows change history

6. **Performance Optimization** ⚡
   - Continuous monitoring and tuning
   - Displays performance metrics

7. **Workflow Integration** 🔌
   - Seamless Claude integration
   - Shows integration status

8. **Pattern Recognition** 🔍
   - Code pattern analysis
   - Displays recognized patterns

### Visual Features
- 🎨 Glass morphism UI design
- 📊 Animated charts and graphs
- 🌐 3D network visualizations
- ⚡ Real-time updates via WebSocket
- 📱 Responsive design

## 🔍 Troubleshooting

### If data doesn't load:
1. Check that the API is running:
   ```bash
   curl http://192.168.1.25:3000/health
   ```

2. Verify the API key is set correctly:
   - Open browser console (F12)
   - Type: `localStorage.getItem('knowledgehub_settings')`
   - Should show the settings with API key

3. Check for errors in console:
   - Look for red error messages
   - Check network tab for failed requests

### Common Issues:
- **CORS errors**: The frontend is configured to handle LAN access
- **401 Unauthorized**: Re-run the localStorage command
- **Empty data**: The system needs time to accumulate data

## 📱 Mobile Access
The Web UI is responsive and can be accessed from mobile devices on the same LAN:
- iPhone/iPad: http://192.168.1.25:3100
- Android: http://192.168.1.25:3100

## 🚀 Quick Test
To verify everything is working, check these endpoints:
- Health: http://192.168.1.25:3000/health
- AI Status: http://192.168.1.25:3000/api/monitoring/ai-features/status
- Web UI: http://192.168.1.25:3100/ai

## 💡 Tips
- The UI updates in real-time as new data comes in
- Use the Settings page to customize the appearance
- All 8 AI features are fully integrated with the backend
- Data persists across sessions thanks to the memory system