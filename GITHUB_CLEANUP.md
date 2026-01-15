# 🔧 GitHub Repository Cleanup Guide
## For: studymate-backend repository

**Repository:** https://github.com/Adi51244/studymate-backend  
**Date:** January 16, 2026

---

## 🚨 EXPOSED FILES IN YOUR GITHUB REPO

These files are currently visible in your GitHub repository and contain sensitive credentials:

1. ✅ `.env` - Contains ALL your API keys (Supabase, Gemini, Firebase)
2. ✅ `.env.backup` - Duplicate with same keys (NOW DELETED locally)
3. ✅ `firebase-service-account.json` - Firebase admin private key

---

## ⚡ STEP 1: Remove Sensitive Files from GitHub (DO THIS NOW)

Open PowerShell and run these commands:

```powershell
# Navigate to your backend folder
cd "d:\Programs\Material\backend"

# Check Git status
git status

# Remove sensitive files from Git tracking (keeps your local .env)
git rm --cached .env
git rm --cached .env.backup
git rm --cached firebase-service-account.json

# Commit the removal
git commit -m "Remove sensitive credentials from version control [SECURITY FIX]"

# Push to GitHub
git push origin main
```

**Note:** If your branch is named `master` instead of `main`, use `git push origin master`

---

## 🔒 STEP 2: Verify .gitignore is Working

```powershell
# Should output the filename if properly ignored
git check-ignore .env
git check-ignore firebase-service-account.json

# Check what Git will commit (should be clean or only show .gitignore)
git status
```

---

## 🔑 STEP 3: Rotate ALL Compromised API Keys

Your exposed keys are **permanently compromised** because they're in Git history. You MUST rotate them:

### A. Google Gemini API Key (CRITICAL - Causing API Violations)
**Current Key:** `AIzaSyB1PE3oJRj4VarXlapj4GSYJaHn0DGTjFc`

1. Go to: https://console.cloud.google.com/apis/credentials
2. Find the compromised key
3. Click **Delete**
4. Click **Create Credentials → API Key**
5. Click **Edit API Key** on the new key
6. Add restrictions:
   - **Application restrictions:** HTTP referrers or IP addresses
   - **API restrictions:** Select only "Generative Language API"
7. Copy new key and update your local `.env` file

### B. Supabase Service Role Key (CRITICAL - Database Admin Access)
**Current Project:** `kcjixqlmxahlmzhrykbl.supabase.co`

1. Go to: https://app.supabase.com/project/kcjixqlmxahlmzhrykbl/settings/api
2. Under **Service Role Key**, click **Reset**
3. Confirm reset
4. Copy new key and update your local `.env` file

### C. Firebase Service Account (CRITICAL - Full Admin Access)
**Current Project:** `pdf-study-assistant`

1. Go to: https://console.firebase.google.com/project/pdf-study-assistant/settings/serviceaccounts
2. Click **Manage service account permissions** (opens Google Cloud Console)
3. Find: `firebase-adminsdk-fbsvc@pdf-study-assistant.iam.gserviceaccount.com`
4. Click the three dots → **Delete**
5. Go back to Firebase Console → **Generate new private key**
6. Download the new JSON file
7. Save it as `firebase-service-account.json` in your backend folder
8. **DO NOT commit this file to Git** (already in .gitignore)

---

## 🧹 STEP 4: Clean Git History (IMPORTANT!)

Even after removing the files, they're still in Git history. Anyone who cloned your repo has them.

### Option A: Using git filter-repo (Recommended)

```powershell
# Install git-filter-repo
pip install git-filter-repo

# Navigate to backend folder
cd "d:\Programs\Material\backend"

# BACKUP FIRST!
cd ..
Copy-Item -Recurse "backend" "backend-backup"
cd backend

# Remove sensitive files from entire history
git filter-repo --invert-paths --path .env --path .env.backup --path firebase-service-account.json --force

# Force push (WARNING: This rewrites history)
git push origin --force --all
```

### Option B: Using BFG Repo Cleaner

```powershell
# Download BFG from: https://rtyley.github.io/bfg-repo-cleaner/

# Clone a mirror
cd "d:\Programs"
git clone --mirror https://github.com/Adi51244/studymate-backend.git

# Clean sensitive files
java -jar bfg.jar --delete-files .env studymate-backend.git
java -jar bfg.jar --delete-files firebase-service-account.json studymate-backend.git

# Clean up
cd studymate-backend.git
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push
git push --force
```

### Option C: Start Fresh (Easiest)

If you don't have important Git history to preserve:

```powershell
# 1. Create NEW repository on GitHub called "studymate-backend-new"

# 2. In your backend folder:
cd "d:\Programs\Material\backend"

# 3. Remove old Git history
Remove-Item -Recurse -Force .git

# 4. Initialize fresh repository
git init
git add .gitignore
git add .env.example
git add *.py
git add requirements.txt
git add Procfile
git add build.sh

# 5. Verify sensitive files are NOT staged
git status
# Should NOT see: .env, firebase-service-account.json

# 6. Commit and push to new repo
git commit -m "Initial commit with secure configuration"
git branch -M main
git remote add origin https://github.com/Adi51244/studymate-backend-new.git
git push -u origin main

# 7. Delete old repository from GitHub
# Go to: https://github.com/Adi51244/studymate-backend/settings
# Scroll to bottom → Delete this repository
```

---

## ✅ VERIFICATION CHECKLIST

After completing all steps:

- [ ] `.env.backup` deleted locally
- [ ] Sensitive files removed from Git tracking
- [ ] `.gitignore` updated
- [ ] Changes pushed to GitHub
- [ ] Verified files not visible on GitHub
- [ ] Gemini API key rotated
- [ ] Supabase service role key rotated
- [ ] Firebase service account recreated
- [ ] Local `.env` updated with new keys
- [ ] Git history cleaned (optional but recommended)
- [ ] Application tested with new credentials
- [ ] No errors in logs

---

## 📋 WHAT'S IN YOUR .env FILE (Keep This Locally)

Your `.env` file should contain:

```env
# Supabase Configuration
SUPABASE_URL=https://kcjixqlmxahlmzhrykbl.supabase.co
SUPABASE_ANON_KEY=<your_anon_key>
SUPABASE_SERVICE_ROLE_KEY=<NEW_KEY_AFTER_ROTATION>

# Google Gemini AI Configuration  
GEMINI_API_KEY=<NEW_KEY_AFTER_ROTATION>

# Firebase Configuration
FIREBASE_PROJECT_ID=pdf-study-assistant
FIREBASE_PRIVATE_KEY=<from_new_service_account_json>
FIREBASE_CLIENT_EMAIL=<from_new_service_account_json>
# ... other Firebase vars

# Google OAuth
GOOGLE_OAUTH_CLIENT_ID=951926380222-nst92boi27ngggnv9t2kds78qrtsntve.apps.googleusercontent.com
```

**NEVER commit this file to Git!**

---

## 🚨 IMMEDIATE ACTIONS SUMMARY

1. **Right Now (5 minutes):**
   ```powershell
   cd "d:\Programs\Material\backend"
   git rm --cached .env .env.backup firebase-service-account.json
   git commit -m "Remove sensitive files [SECURITY]"
   git push origin main
   ```

2. **Within 1 Hour:**
   - Rotate Gemini API key
   - Reset Supabase service role key
   - Recreate Firebase service account

3. **Within 24 Hours:**
   - Clean Git history (or create new repo)
   - Test application with new credentials
   - Monitor for unauthorized usage

---

## 📞 NEED HELP?

If you see unauthorized activity:
- **Supabase Support:** support@supabase.com
- **Firebase Support:** https://firebase.google.com/support
- **Google Cloud Support:** https://cloud.google.com/support

---

**Time Required:** 
- Removing files: 5 minutes
- Rotating keys: 30-45 minutes
- Cleaning history: 15 minutes
- **Total: ~1 hour**

🚀 **Start with the PowerShell commands above RIGHT NOW!**
