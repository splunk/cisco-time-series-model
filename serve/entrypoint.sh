#!/bin/sh
set -e

# Fix ownership of mounted volumes (root-owned on first creation).
chown -R appuser:appuser /home/appuser/.cache 2>/dev/null || true

exec gosu appuser "$@"
