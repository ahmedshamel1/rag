#!/bin/bash

# Docker Setup Script for RAG Application in WSL2
# This script helps set up and manage the Docker environment

set -e

echo "🐳 Docker Setup Script for RAG Application"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop or Docker service."
    echo "💡 In WSL2, make sure Docker Desktop is running on Windows."
    exit 1
fi

echo "✅ Docker is running"

# Function to build and start the application
build_and_start() {
    echo "🔨 Building Docker images..."
    docker-compose build
    
    echo "🚀 Starting services..."
    docker-compose up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 10
    
    echo "🔍 Checking service health..."
    docker-compose ps
    
    echo "🌐 Backend: http://localhost:8000"
    echo "🌐 Frontend: http://localhost:8501"
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping services..."
    docker-compose down
    echo "✅ Services stopped"
}

# Function to view logs
view_logs() {
    echo "📋 Viewing logs... (Ctrl+C to exit)"
    docker-compose logs -f
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    echo "✅ Cleanup completed"
}

# Main menu
case "${1:-}" in
    "build")
        build_and_start
        ;;
    "stop")
        stop_services
        ;;
    "logs")
        view_logs
        ;;
    "cleanup")
        cleanup
        ;;
    "restart")
        stop_services
        sleep 2
        build_and_start
        ;;
    *)
        echo "Usage: $0 {build|stop|logs|cleanup|restart}"
        echo ""
        echo "Commands:"
        echo "  build    - Build and start services (frontend + backend)"
        echo "  stop     - Stop all services"
        echo "  logs     - View service logs"
        echo "  cleanup  - Clean up Docker resources"
        echo "  restart  - Restart all services"
        echo ""
        echo "Examples:"
        echo "  $0 build     # Start both frontend and backend"
        echo "  $0 logs      # View logs"
        exit 1
        ;;
esac
