echo "Current branch: $(git branch --show-current)"

echo "Pulling latest changes"
git pull

echo "Updating submodules and applying patches"

git submodule sync
git submodule update --init --recursive

PATCH="../../patches/browser-use.patch"
SUBMODULE="external/browser-use"

if git -C "$SUBMODULE" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "Patch already applied, skipping."
else
    git -C "$SUBMODULE" apply "$PATCH"
fi