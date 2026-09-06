#!/usr/bin/env bash
# Regenerate web/styles.css from web/src/input.css.
#
# Toolchain (a ~110 MB standalone Tailwind CLI + the DaisyUI package tarball) is
# fetched with curl into web/.cache/ (gitignored). No npm / node required.
# The built web/styles.css IS committed, so GitHub Pages serves it directly with
# no build step - same policy as the generated figures in web/assets/.
set -euo pipefail
cd "$(dirname "$0")"

TW_VERSION="v4.3.3"
DAISY_VERSION="5.7.28"
CACHE=".cache"
mkdir -p "$CACHE"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  TW_ASSET="tailwindcss-linux-x64" ;;
  Linux-aarch64) TW_ASSET="tailwindcss-linux-arm64" ;;
  Darwin-x86_64) TW_ASSET="tailwindcss-macos-x64" ;;
  Darwin-arm64)  TW_ASSET="tailwindcss-macos-arm64" ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

TW="$CACHE/tailwindcss"
if [ ! -x "$TW" ]; then
  echo "fetching tailwindcss $TW_VERSION ($TW_ASSET) ..."
  curl -fsSL -o "$TW" \
    "https://github.com/tailwindlabs/tailwindcss/releases/download/$TW_VERSION/$TW_ASSET"
  chmod +x "$TW"
fi

if [ ! -f "$CACHE/daisyui/index.js" ]; then
  echo "fetching daisyui $DAISY_VERSION ..."
  curl -fsSL -o "$CACHE/daisyui.tgz" \
    "https://registry.npmjs.org/daisyui/-/daisyui-$DAISY_VERSION.tgz"
  rm -rf "$CACHE/daisyui" && mkdir -p "$CACHE/daisyui"
  tar -xzf "$CACHE/daisyui.tgz" -C "$CACHE/daisyui" --strip-components=1
  rm "$CACHE/daisyui.tgz"
fi

"$TW" -i src/input.css -o styles.css --minify
echo "wrote web/styles.css ($(wc -c < styles.css) bytes)"
