# Docker Setup for RAG Application

This document provides instructions for running the RAG application using Docker and Docker Compose.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB of available RAM
- Ports 8000 and 8501 available

### One-Command Setup
```bash
cd rag
docker-compose up --build
```

This will:
- Build both backend and frontend Docker images
- Start the FastAPI backend on port 8000
- Start the Streamlit frontend on port 8501
- Create a persistent ChromaDB volume for data storage

## Service Endpoints

- **Backend API**: http://localhost:8000/
- **Frontend Chat**: http://localhost:8501/
- **API Health Check**: http://localhost:8000/

## Testing the Setup

### 1. Verify Services are Running
```bash
docker-compose ps
```

Both services should show "Up" status.

### 2. Test Endpoints
```bash
# Test backend
curl http://localhost:8000/

# Test frontend
curl http://localhost:8000/
```

### 3. Verify ChromaDB Persistence
1. Add a document to the `assests/` folder
2. Check that the document is processed and stored
3. Stop services: `docker-compose down`
4. Restart services: `docker-compose up`
5. Verify the document data persists

## Available Commands

### Using Make
```bash
make build      # Build and start services
make up         # Start services
make down       # Stop services
make logs       # View logs
make clean      # Clean up Docker resources
make restart    # Restart services
make test       # Run health checks
```

### Using Docker Compose Directly
```bash
docker-compose up --build    # Build and start
docker-compose up -d         # Start in background
docker-compose down          # Stop services
docker-compose logs -f       # Follow logs
docker-compose ps            # Check status
```

### Using the Setup Script
```bash
./scripts/docker-setup.sh build    # Build and start
./scripts/docker-setup.sh stop     # Stop services
./scripts/docker-setup.sh logs     # View logs
./scripts/docker-setup.sh cleanup  # Clean up
```

## Data Persistence

### Production/Test Mode (Default)
- ChromaDB data is stored in a named Docker volume `chroma_data`
- Data persists across container restarts and rebuilds
- Volume is automatically created and managed

### Developer Mode
To use bind mounts for development:

1. Copy the override file:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

2. Edit `docker-compose.override.yml` and uncomment the bind mount sections

3. Restart services:
```bash
docker-compose down
docker-compose up --build
```

## Backup and Restore

### Backup ChromaDB Data
```bash
make backup
# or
docker run --rm -v rag_chroma_data:/data -v $(pwd):/backup alpine tar czf /backup/chroma_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
```

### Restore ChromaDB Data
```bash
make restore BACKUP_FILE=chroma_backup_20241201_143022.tar.gz
# or
docker run --rm -v rag_chroma_data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/chroma_backup_20241201_143022.tar.gz --strip-components=1"
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the ports
   netstat -tulpn | grep :8000
   netstat -tulpn | grep :8501
   
   # Stop conflicting services or change ports in docker-compose.yml
   ```

2. **Permission Issues**
   ```bash
   # Fix database permissions
   docker-compose exec backend chown -R app:app /app/database
   ```

3. **Out of Memory**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # or use swap space
   ```

4. **Service Won't Start**
   ```bash
   # Check logs
   docker-compose logs backend
   docker-compose logs frontend
   
   # Check health status
   docker-compose ps
   ```

### Health Checks
Both services include health checks that verify they're responding:
- Backend: HTTP GET to `/`
- Frontend: HTTP GET to `/`

Health checks run every 30 seconds and will restart unhealthy containers.

## Development Workflow

### Live Code Changes
For development with live code reloading:

1. Use the override file with bind mounts
2. Mount source code directories
3. Restart services when making changes

### Adding New Dependencies
1. Update the appropriate `requirements.txt` file
2. Rebuild the Docker image:
   ```bash
   docker-compose build --no-cache
   docker-compose up
   ```

## Performance Considerations

- **Memory**: Each service uses ~1-2GB RAM
- **Storage**: ChromaDB volume can grow large with many documents
- **CPU**: Document processing is CPU-intensive
- **Network**: Services communicate via Docker internal network

## Security Notes

- Services run as non-root users
- Database directory has restricted permissions
- Health checks use internal endpoints only
- No sensitive data in Docker images

## Cleanup

To completely remove all Docker resources:
```bash
make clean
# or
docker-compose down --volumes --remove-orphans
docker system prune -f
```

**Warning**: This will delete all ChromaDB data. Use `make backup` first if you need to preserve data.
