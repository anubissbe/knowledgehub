# Build stage
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY src/web-ui/package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src/web-ui ./

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration if exists
# COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]