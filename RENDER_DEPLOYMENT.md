# Deploying AI Agent to Render via Docker Hub

This guide walks through building your Docker image, pushing it to Docker Hub, and deploying it as a container on Render.

## Step 1: Build Your Docker Image

```bash
# Navigate to your project directory
cd /path/to/agent-server

# Build the Docker image with a tag that includes your Docker Hub username
docker build -t yourusername/ai-agent:latest .

# If you're on Apple Silicon (M1/M2/M3) and want to build for multiple platforms
docker buildx create --name mybuilder --use
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/ai-agent:latest --push .
```

## Step 2: Test Your Docker Image Locally

```bash
# Run the container locally to test
docker run -p 10000:10000 -e GEMINI_API_KEY=your_gemini_api_key yourusername/ai-agent:latest

# Access the API at http://localhost:10000
```

## Step 3: Push to Docker Hub

```bash
# Log in to Docker Hub
docker login

# Push your image to Docker Hub (if not using buildx with --push flag)
docker push yourusername/ai-agent:latest

# Note: If you used buildx with the --push flag in Step 1, you can skip this step
# as the image is already pushed to Docker Hub
```

## Step 4: Deploy on Render

1. **Create a new Web Service on Render**:
   - Sign in to your Render account
   - Click "New" and select "Web Service"

2. **Choose Docker Hub deployment**:
   - Select "Deploy an existing image from a registry"
   - Enter your Docker Hub image: `yourusername/ai-agent:latest`

3. **Configure the service**:
   - **Name**: `ai-agent` (or your preferred name)
   - **Environment Variables**:
     - `GEMINI_API_KEY`: Your Gemini API key

4. **Advanced Settings (optional)**:
   - Configure health check path: `/docs` (FastAPI auto-generated docs)

5. **Create Web Service**:
   - Click "Create Web Service"

## Step 5: Verify Deployment

1. Once Render completes the deployment, click on the generated URL to access your API.

2. You can check the API documentation at `https://your-render-url.onrender.com/docs`

## Updating Your Deployment

When you make changes to your application:

1. Build a new Docker image:
   ```bash
   docker build -t yourusername/ai-agent:latest .
   ```

2. Push the updated image to Docker Hub:
   ```bash
   docker push yourusername/ai-agent:latest
   ```

3. On Render:
   - Either set up auto-deploy from Docker Hub (in service settings)
   - Or manually deploy the latest version from the Render dashboard

## Notes

- Render will automatically assign a `PORT` environment variable that your container should use.
- Your application will be accessible via HTTPS through the Render-assigned domain.
- Consider setting up CI/CD workflows for automated deployments.
- This deployment assumes MCP servers are configured correctly. The Docker image doesn't install Node.js because your local environment doesn't require it. If you encounter MCP-related issues, check the server configuration in `client/server_config.json`.

## Troubleshooting

### Docker Build Issues

- **Build Cache Issues**: If you encounter issues with the build, try using the `--no-cache` flag:
  ```bash
  docker build --no-cache -t yourusername/ai-agent:latest .
  ```

- **Architecture Issues**: If deploying to a different architecture than your build machine:
  ```bash
  # For multi-architecture builds (recommended for Render)
  docker buildx build --platform linux/amd64 -t yourusername/ai-agent:latest --push .
  ```

- **MCP Server Configuration**: If your application needs specific MCP servers that aren't working in the Docker container, check the paths in `client/server_config.json` and make any necessary adjustments.

### Render Deployment Issues

- If you encounter container startup failures on Render, check the logs in the Render dashboard.
- Ensure all environment variables (especially `GEMINI_API_KEY`) are correctly set in Render.
- Verify that your container is listening on the port specified by Render's `PORT` environment variable.
