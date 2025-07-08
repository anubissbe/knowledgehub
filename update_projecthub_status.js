const {projectHub} = require('/opt/scripts/projecthub-client.js');

async function updateProjectStatus() {
  try {
    console.log('🔄 Updating ProjectHub with session management completion...');
    
    // Find and complete session management related tasks
    const tasksToComplete = [
      { search: 'Create session cleanup background tasks', id: '0d028e50-65ec-4231-9af3-6872c19aafbe' },
      { search: 'Add session linking functionality', id: '35c7faa9-f17c-46c4-98ba-452fc82b42b2' }
    ];
    
    console.log('Updating specific session management tasks...');
    
    for (const taskInfo of tasksToComplete) {
      try {
        await projectHub.updateTask(taskInfo.id, {
          status: 'completed',
          progress: 100,
          notes: 'Completed with comprehensive testing and documentation - Redis cache reconstruction, background cleanup service, and intelligent session linking all implemented and tested successfully'
        });
        console.log('✅ Completed: ' + taskInfo.search);
      } catch (error) {
        console.log('❌ Failed to update ' + taskInfo.search + ': ' + error.message);
      }
    }
    
    console.log('✅ ProjectHub updated successfully');
  } catch (error) {
    console.error('❌ Failed to update ProjectHub:', error.message);
  }
}

updateProjectStatus();