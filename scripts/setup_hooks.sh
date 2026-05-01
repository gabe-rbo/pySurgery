#!/bin/bash
# Setup script to install the interactive version-bumping git hook

HOOK_PATH=".git/hooks/pre-push"

cat << 'EOF' > "$HOOK_PATH"
#!/bin/bash
# Interactive Version Bumper for pySurgery
# Installed as .git/hooks/pre-push

LAST_MSG=$(git log -1 --pretty=%B)
if [[ "$LAST_MSG" =~ "Bump version" ]]; then
    exit 0
fi

exec < /dev/tty

CUR_VERSION=$(python3 -c "import configparser; config = configparser.ConfigParser(); config.read('.bumpversion.cfg'); print(config['bumpversion']['current_version'])")
NEXT_PATCH=$(python3 -c "v = '$CUR_VERSION'.split('.'); v[2] = str(int(v[2]) + 1); print('.'.join(v))")
NEXT_MINOR=$(python3 -c "v = '$CUR_VERSION'.split('.'); v[1] = str(int(v[1]) + 1); v[2] = '0'; print('.'.join(v))")
NEXT_MAJOR=$(python3 -c "v = '$CUR_VERSION'.split('.'); v[0] = str(int(v[0]) + 1); v[1] = '0'; v[2] = '0'; print('.'.join(v))")

echo ""
echo "----------------------------------------------------"
echo "🚀 PY-SURGERY VERSION MANAGER (Current: v$CUR_VERSION)"
echo "----------------------------------------------------"
read -p "Would you like to bump the version before pushing? (y/N): " WANT_BUMP

if [[ "$WANT_BUMP" =~ ^[Yy]$ ]]; then
    echo "Select bump level:"
    echo "  1) Patch: v$CUR_VERSION -> v$NEXT_PATCH (Bug fixes)"
    echo "  2) Minor: v$CUR_VERSION -> v$NEXT_MINOR (New features)"
    echo "  3) Major: v$CUR_VERSION -> v$NEXT_MAJOR (Breaking changes)"
    read -p "Choice [1-3, default=1]: " CHOICE

    case "$CHOICE" in
        2) BUMP="minor" ;;
        3) BUMP="major" ;;
        *) BUMP="patch" ;;
    esac

    if ! command -v bump-my-version &> /dev/null; then
        echo "❌ Error: bump-my-version is not installed. Run 'pip install bump-my-version'."
        exit 1
    fi

    if bump-my-version bump "$BUMP"; then
        echo "✅ Success. Pushing new version and tags..."
        git push --follow-tags
        exit 1 
    else
        echo "❌ Bump failed."
    fi
fi
echo "Skipping version bump."
echo "----------------------------------------------------"
echo ""
EOF

chmod +x "$HOOK_PATH"
echo "Interactive Git pre-push hook installed successfully."
