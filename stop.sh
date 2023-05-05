#!/bin/bash
pids=$(ps aux | grep -E 'replit-code-v1-3b-server|python|gunicorn.*flask_app' | grep -v grep | awk '{print $2}')
if [ -n "$pids" ]; then
    echo "Killing processes: $pids"
    kill $pids
else
    echo "No matching processes found"
fi
