# Publishing to GitHub - Step by Step Guide

This guide will help you publish your LSTM Time Series project to GitHub.

## Prerequisites

- Git installed on your machine
- GitHub account created
- GitHub CLI (optional but recommended)

## Step 1: Initialize Git Repository (Local)

```bash
# Navigate to your project directory
cd /Applications/CODES

# Initialize git repository
git init

# Add all files to staging
git add LSTM-TimeSeries.ipynb README.md requirements.txt LICENSE .gitignore

# Create initial commit
git commit -m "Initial commit: LSTM Time Series Forecasting implementation"
```

## Step 2: Create GitHub Repository

### Option A: Using GitHub Website (Recommended for beginners)

1. Go to [GitHub](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `LSTM-TimeSeries`
   - **Description**: "Time series forecasting using LSTM neural networks on airline passengers dataset"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

### Option B: Using GitHub CLI

```bash
# Create a new repository on GitHub
gh repo create LSTM-TimeSeries --public --source=. --remote=origin --push

# This will:
# - Create the repository on GitHub
# - Add it as remote 'origin'
# - Push your code
```

## Step 3: Connect Local Repository to GitHub

If you used Option A, run these commands (GitHub will show them after creating the repo):

```bash
# Add GitHub repository as remote
git remote add origin https://github.com/jugalmodi0111/LSTM-TimeSeries.git

# Verify remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify Your Repository

1. Go to `https://github.com/jugalmodi0111/LSTM-TimeSeries`
2. Check that all files are present:
   - ✅ LSTM-TimeSeries.ipynb
   - ✅ README.md
   - ✅ requirements.txt
   - ✅ LICENSE
   - ✅ .gitignore

## Step 5: Enhance Your Repository (Optional)

### Add Topics/Tags

1. Go to your repository page
2. Click the gear icon next to "About"
3. Add topics: `machine-learning`, `deep-learning`, `lstm`, `time-series`, `keras`, `tensorflow`, `python`, `jupyter-notebook`

### Add Repository Description

In the same "About" section, add:
- **Description**: "Time series forecasting using LSTM neural networks on airline passengers dataset"
- **Website**: (optional) Link to your portfolio or blog

### Enable GitHub Pages (to view notebook)

1. Go to Settings → Pages
2. Select source: Deploy from branch
3. Choose branch: `main` and folder: `/ (root)`
4. Your notebook will be viewable (though not rendered) at the GitHub Pages URL

### Add Badges to README

The README already includes badges for Python, Keras, and License.

## Step 6: Future Updates

When you make changes to your code:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit with descriptive message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Quick Command Reference

```bash
# Initialize and first push
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/jugalmodi0111/LSTM-TimeSeries.git
git push -u origin main

# Subsequent updates
git add .
git commit -m "Update description"
git push
```

## Troubleshooting

### Issue: Authentication Failed

**Solution**: Use a Personal Access Token (PAT) instead of password
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

Or use SSH keys:
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

### Issue: Large Files

If you have large model files (*.h5, *.pkl):
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.pkl"

# Add and commit
git add .gitattributes
git commit -m "Configure Git LFS"
```

## Next Steps

1. ⭐ Add a description to your repository
2. 📝 Consider adding a CONTRIBUTING.md if you want contributions
3. 🏷️ Add topics/tags for discoverability
4. 📊 Add screenshots or GIFs of the visualization to README
5. 🔗 Share your repository on social media or forums

## Example Repository Structure

```
LSTM-TimeSeries/
├── .git/                      # Git directory (hidden)
├── .gitignore                 # Files to ignore
├── LSTM-TimeSeries.ipynb      # Main notebook
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
└── GITHUB_PUBLISH_GUIDE.md    # This guide
```

## Useful GitHub Features

- **Issues**: Track bugs and feature requests
- **Projects**: Organize work with kanban boards
- **Actions**: Set up CI/CD pipelines
- **Wiki**: Additional documentation
- **Discussions**: Community forum for your project

---

**Ready to publish?** Start with Step 1 above! 🚀
