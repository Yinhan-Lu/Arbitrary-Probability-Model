#!/bin/bash

# Helper script for pushing changes to GitHub
# Ensures all requirements are met before pushing

set -e

echo "======================================================================"
echo "Git Push Helper - GPT-2 PyTorch Project"
echo "======================================================================"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes."
    echo ""
    git status --short
    echo ""
    read -p "Do you want to commit these changes? (y/n): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Enter commit message (in English):"
        read -r commit_message

        if [ -z "$commit_message" ]; then
            echo "Error: Commit message cannot be empty"
            exit 1
        fi

        # Check if message is in English (basic check)
        if echo "$commit_message" | grep -qP '[\x{4e00}-\x{9fff}]'; then
            echo "❌ Error: Commit message contains Chinese characters"
            echo "Please use English only for all commits"
            exit 1
        fi

        echo ""
        echo "Adding files..."
        git add .

        echo "Committing with message: $commit_message"
        git commit -m "$commit_message"
    else
        echo "Aborted. Please commit your changes manually."
        exit 1
    fi
fi

# Security checks
echo ""
echo "Running security checks..."
echo "======================================================================"

# Check for large files (>10MB)
large_files=$(find . -type f -size +10M 2>/dev/null | grep -v ".git" | grep -v "logs/" | grep -v "checkpoints/" || true)
if [ -n "$large_files" ]; then
    echo "⚠️  Warning: Large files detected (>10MB):"
    echo "$large_files"
    echo ""
    read -p "These files might be too large for GitHub. Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please review large files."
        exit 1
    fi
fi

# Check for potential checkpoint files
checkpoint_files=$(git ls-files | grep -E '\.(pt|pth|ckpt)$' || true)
if [ -n "$checkpoint_files" ]; then
    echo "❌ Error: Checkpoint files detected in git:"
    echo "$checkpoint_files"
    echo ""
    echo "These should not be committed. Please remove them:"
    echo "  git rm --cached <file>"
    exit 1
fi

# Check for potential credential files
credential_patterns="\.env|secrets|credentials|api_key|\.key|\.pem"
cred_files=$(git ls-files | grep -E "$credential_patterns" || true)
if [ -n "$cred_files" ]; then
    echo "⚠️  Warning: Potential credential files detected:"
    echo "$cred_files"
    echo ""
    read -p "Are you sure these don't contain secrets? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please review credential files."
        exit 1
    fi
fi

echo "✓ Security checks passed"
echo ""

# Show what will be pushed
echo "======================================================================"
echo "Commits to be pushed:"
echo "======================================================================"
git log origin/main..HEAD --oneline 2>/dev/null || git log --oneline -n 5

echo ""
echo "======================================================================"
echo "Ready to push to GitHub"
echo "======================================================================"
read -p "Push to origin/main? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Pushing to GitHub..."
    git push origin main

    echo ""
    echo "======================================================================"
    echo "✓ Successfully pushed to GitHub!"
    echo "======================================================================"
    echo ""
    echo "Remember:"
    echo "  - All content should be in English"
    echo "  - No sensitive data committed"
    echo "  - Repository is private"
    echo ""
else
    echo "Push cancelled."
    exit 1
fi
