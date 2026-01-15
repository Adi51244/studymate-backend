# ✅ Quick Checklist - Fix Your GitHub Repo

## RIGHT NOW - Run These Commands:

```powershell
cd "d:\Programs\Material\backend"
git rm --cached .env
git rm --cached .env.backup
git rm --cached firebase-service-account.json
git commit -m "Remove sensitive credentials [SECURITY]"
git push origin main
```

## Then Rotate These Keys:

### 1. Gemini API (URGENT - Causing your API violations)
- Go to: https://console.cloud.google.com/apis/credentials
- Delete key: AIzaSyB1PE3oJRj4VarXlapj4GSYJaHn0DGTjFc
- Create new key with restrictions
- Update `.env` file

### 2. Supabase Service Role Key (URGENT)
- Go to: https://app.supabase.com/project/kcjixqlmxahlmzhrykbl/settings/api
- Click "Reset" on service_role key
- Copy new key to `.env`

### 3. Firebase Service Account (URGENT)
- Go to: https://console.firebase.google.com/project/pdf-study-assistant/settings/serviceaccounts
- Delete old service account
- Create new one
- Download new JSON file
- Save as `firebase-service-account.json`

## Files Status:
✅ `.env.backup` - DELETED (done)
✅ `.gitignore` - UPDATED (done)
✅ Keeping only one `.env` file (LOCAL ONLY, not on GitHub)

## Read Full Instructions:
See `GITHUB_CLEANUP.md` for complete step-by-step guide.
