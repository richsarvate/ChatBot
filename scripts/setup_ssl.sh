#!/bin/bash
# Generate self-signed SSL certificates for development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CERTS_DIR="$PROJECT_ROOT/certs"

# Create certs directory if it doesn't exist
mkdir -p "$CERTS_DIR"

# Check if certificates already exist
if [ -f "$CERTS_DIR/cert.pem" ] && [ -f "$CERTS_DIR/key.pem" ]; then
    echo "SSL certificates already exist."
    read -p "Do you want to regenerate them? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing certificates."
        exit 0
    fi
fi

echo "Generating self-signed SSL certificate..."
echo "This certificate is for development use only."
echo

# Get domain/hostname (default to localhost)
read -p "Enter hostname (default: localhost): " HOSTNAME
HOSTNAME=${HOSTNAME:-localhost}

# Generate certificate
openssl req -x509 -newkey rsa:4096 -nodes \
    -out "$CERTS_DIR/cert.pem" \
    -keyout "$CERTS_DIR/key.pem" \
    -days 365 \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=$HOSTNAME" \
    2>/dev/null

# Set appropriate permissions
chmod 600 "$CERTS_DIR/key.pem"
chmod 644 "$CERTS_DIR/cert.pem"

echo
echo "✓ SSL certificates generated successfully!"
echo "  Certificate: $CERTS_DIR/cert.pem"
echo "  Private key: $CERTS_DIR/key.pem"
echo
echo "Note: This is a self-signed certificate. Your browser will show a warning."
echo "For production use, obtain certificates from a trusted Certificate Authority."
echo
echo "To start the server with HTTPS, run: bash scripts/start.sh"
