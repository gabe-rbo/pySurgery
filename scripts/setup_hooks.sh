#!/bin/bash
# Setup script to install the pySurgery versioning hooks

# 1. Global Git Config for pySurgery
git config push.followTags true

# 2. Pre-commit Hook (Interactive Bump with Intent Tracking)
cat << 'EOF' > .git/hooks/pre-commit
#!/bin/bash
exec < /dev/tty
CUR_VERSION=$(python3 -c "import configparser; config = configparser.ConfigParser(); config.read('.bumpversion.cfg'); print(config['bumpversion']['current_version'])")
NEXT_PATCH=$(python3 -c "v = '$CUR_VERSION'.split('.'); v[2] = str(int(v[2]) + 1); print('.'.join(v))")
NEXT_MINOR=$(python3 -c "v = '$CUR_VERSION'.split('.'); v[1] = str(int(v[1]) + 1); v[2] = '0'; print('.'.join(v))")
NEXT_MAJOR=$(python3 -c "v = '$CUR_VERSION'.split('.'); v[0] = str(int(v[0]) + 1); v[1] = '0'; v[2] = '0'; print('.'.join(v))")

echo ""
echo "----------------------------------------------------"
read -p "Do you want to update the packages version? (Current version: $CUR_VERSION) [y/N]: " WANT_BUMP

if [[ "$WANT_BUMP" =~ ^[Yy]$ ]]; then
    echo "Select bump level:"
    echo "  1) Patch: $CUR_VERSION -> $NEXT_PATCH"
    echo "  2) Minor: $CUR_VERSION -> $NEXT_MINOR"
    echo "  3) Major: $CUR_VERSION -> $NEXT_MAJOR"
    read -p "Choice [1-3, default=1]: " CHOICE
    case "$CHOICE" in
        2) BUMP="minor" ;;
        3) BUMP="major" ;;
        *) BUMP="patch" ;;
    esac
    if bump-my-version bump "$BUMP" --no-commit --no-tag --allow-dirty; then
        git add pyproject.toml .bumpversion.cfg CITATION.cff
        FINAL_VERSION=$(python3 -c "import configparser; config = configparser.ConfigParser(); config.read('.bumpversion.cfg'); print(config['bumpversion']['current_version'])")
        echo "v$FINAL_VERSION" > .git/PY_SURGERY_RELEASE_PENDING
        echo "✅ Version updated to v$FINAL_VERSION. It will be released on push."
    else
        echo "❌ Bump failed."
        exit 1
    fi
fi
echo "----------------------------------------------------"
echo ""
EOF

# 3. Pre-push Hook (Intent-Based Tagging)
cat << 'EOF' > .git/hooks/pre-push
#!/bin/bash
PENDING_FILE=".git/PY_SURGERY_RELEASE_PENDING"
if [ -f "$PENDING_FILE" ]; then
    TAG=$(cat "$PENDING_FILE")
    git tag -a "$TAG" -m "Release $TAG"
    rm "$PENDING_FILE"
    echo ">>> New version v$TAG tagged for release."
fi
EOF

chmod +x .git/hooks/pre-commit .git/hooks/pre-push
echo "pySurgery Git hooks (intent-based) installed successfully."
