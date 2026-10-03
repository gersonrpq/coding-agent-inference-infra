#!/usr/bin/env bash

echo "Sync PC → $LAMBDA:~/final_project"
ssh -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=accept-new "$LAMBDA" "mkdir -p ~/final_project"
rsync -avz -e "ssh -i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=accept-new" \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '.git' \
  ./ \
  "$LAMBDA:~/final_project/"
echo "Synced."
