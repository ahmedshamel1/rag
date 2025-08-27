#!/bin/bash
set -e

# Create database directory if it doesn't exist
mkdir -p /app/database

# Fix permissions for the database directory
chown -R app:app /app/database
chmod -R 755 /app/database

# Execute the main command
exec "$@"
