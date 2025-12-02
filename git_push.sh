#!/bin/bash

# Git Push Script
# Author: Yinhan Lu
# Email: yinhan.lu@mail.mcgill.ca

# Configuration
GIT_USER_NAME="Yinhan-Lu"
GIT_USER_EMAIL="yinhan.lu@mail.mcgill.ca"

# Set git user config (local to this repo)
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# Staging mode: default = tracked changes only (avoids sweeping untracked files)
STAGE_ALL=0
if [ "$1" = "--all" ]; then
    STAGE_ALL=1
    shift
fi

# Check if commit message is provided (after optional flag)
if [ -z "$1" ]; then
    echo "Usage: ./git_push.sh [--all] \"your commit message\""
    echo "  --all  Stage untracked files too (default stages only tracked changes)"
    echo "Example: ./git_push.sh \"Add new feature\""
    exit 1
fi

COMMIT_MSG="$1"

# Show current status
echo "=========================================="
echo "Git Push Script"
echo "Author: $GIT_USER_NAME <$GIT_USER_EMAIL>"
echo "=========================================="
echo ""

# Show changes
echo "[1/4] Checking status..."
git status --short
echo ""

# Stage changes
echo "[2/4] Staging changes..."
if [ $STAGE_ALL -eq 1 ]; then
    echo "  • Including untracked files (git add -A)"
    git add -A
else
    echo "  • Tracked changes only (git add -u). Untracked files left untouched."
    git add -u
fi
echo ""

# Commit
echo "[3/4] Committing with message: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"
echo ""

# Push
echo "[4/4] Pushing to remote..."
git push origin $(git branch --show-current)

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
