#!/bin/bash
echo "First arg: $1"
echo "Second arg: $2"
git add .
git commit -m "update"
git push origin main