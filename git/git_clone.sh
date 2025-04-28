#!/bin/bash

# GitHub username and repository name
REPO_NAME=$1
GITHUB_USERNAME=$2
PAT=$3

# Check if Git is installed
if ! [ -x "$(command -v git)" ]; then
  echo "Error: git is not installed." >&2
  exit 1
fi

# Clone the private repository
echo "Cloning the private GitHub repository..."
git clone https://$GITHUB_USERNAME:$PAT@github.com/$REPO_NAME.git

# Check if the clone succeeded
if [ $? -eq 0 ]; then
  echo "Repository cloned successfully."
else
  echo "Failed to clone the repository. Please check your credentials and repository settings."
  exit 1
fi