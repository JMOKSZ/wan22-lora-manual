#!/bin/bash
cd "$(dirname "$0")"
exec /usr/bin/python3 -m http.server 8331 --bind 127.0.0.1
