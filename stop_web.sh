#!/bin/bash
pid=$(ps -ef | grep 'python.*app.py' | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo "Killing process $pid"
    kill $pid
else
    echo "No process found"
fi