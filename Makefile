.PHONY: help up down build logs clean restart backup restore test

# Default target
help:
	@echo "RAG Application Docker Commands"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make up      - Start services (docker-compose up)"
	@echo "  make down    - Stop services (docker-compose down)"
	@echo "  make build   - Build and start services (docker-compose up --build)"
	@echo "  make logs    - View service logs"
	@echo "  make clean   - Clean up Docker resources"
	@echo "  make restart - Restart all services"
	@echo "  make backup  - Backup ChromaDB data"
	@echo "  make restore - Restore ChromaDB data from backup"
	@echo "  make test    - Run quick health checks"
	@echo ""

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# Build and start services
build:
	docker-compose up --build -d

# View logs
logs:
	docker-compose logs -f

# Clean up Docker resources
clean:
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Restart services
restart: down
	sleep 2
	make build

# Backup ChromaDB data
backup:
	@echo "Creating backup of ChromaDB data..."
	docker run --rm -v rag_chroma_data:/data -v $(PWD):/backup alpine tar czf /backup/chroma_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz -C /data .
	@echo "Backup created in current directory"

# Restore ChromaDB data from backup
restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore BACKUP_FILE=filename.tar.gz"; \
		echo "Available backups:"; \
		ls -la chroma_backup_*.tar.gz 2>/dev/null || echo "No backups found"; \
		exit 1; \
	fi
	@echo "Restoring ChromaDB data from $(BACKUP_FILE)..."
	docker run --rm -v rag_chroma_data:/data -v $(PWD):/backup alpine sh -c "cd /data && tar xzf /backup/$(BACKUP_FILE) --strip-components=1"
	@echo "Restore completed"

# Quick health check
test:
	@echo "Running health checks..."
	@echo "Backend: http://localhost:8000"
	@curl -s -f http://localhost:8000/ > /dev/null && echo "✅ Backend is healthy" || echo "❌ Backend is not responding"
	@echo "Frontend: http://localhost:8501"
	@curl -s -f http://localhost:8501/ > /dev/null && echo "✅ Frontend is healthy" || echo "❌ Frontend is not responding"
