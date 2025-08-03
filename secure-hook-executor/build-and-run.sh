#!/bin/bash
# Build and run the secure hook executor

# Build the Docker image
docker build -t knowledgehub-hook-executor .

# Example: Run the hook with maximum security
# This command demonstrates all the security controls from the blueprint
echo "How do I implement authentication in FastAPI?" | docker run \
  --rm \
  --read-only \
  --network=knowledgehub_default \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --memory=100m \
  --cpus=0.5 \
  --user=1000:1000 \
  -e KNOWLEDGEHUB_API_URL=http://api:3000 \
  -e KNOWLEDGEHUB_API_KEY=your-api-key \
  -e CLAUDE_USER_ID=test-user \
  -e CLAUDE_PROJECT_ID=test-project \
  -i knowledgehub-hook-executor