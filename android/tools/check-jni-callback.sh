#!/usr/bin/env bash
#
# Fails if a built APK is missing a method pocket-speak's engine finds by name.
#
# pocket-speak/crates/android resolves these from native code, so nothing in Kotlin
# breaks when they disappear:
#
#   - NativeEngine's native methods, which the loader binds to the exported
#     Java_org_pockettts_android_engine_NativeEngine_* symbols by name;
#   - Sink.onAudio, which the engine calls per chunk with
#     call_method(sink, "onAudio", "([F)Z").
#
# R8 renaming or inlining any of them compiles cleanly and passes every test.
# Under sherpa-onnx its audio callback vanished twice that way, by two
# different mechanisms. The only place the answer exists is the built
# artefact, so that is what this reads. Usage:
#
#   tools/check-jni-callback.sh app/build/outputs/apk/release/app-release.apk
#
set -euo pipefail

apk="${1:?usage: check-jni-callback.sh <apk>}"
[ -f "$apk" ] || { echo "no such APK: $apk" >&2; exit 2; }

sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/android-sdk}}"
dexdump="$(ls -1 "$sdk"/build-tools/*/dexdump 2>/dev/null | sort -V | tail -1 || true)"
[ -x "${dexdump:-}" ] || { echo "dexdump not found under $sdk/build-tools" >&2; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
unzip -q -o "$apk" 'classes*.dex' -d "$work"

# Declarations only, as "name type" pairs: a call site can survive in one dex
# while the method it names is gone from all of them.
for dex in "$work"/classes*.dex; do
    "$dexdump" -d "$dex" 2>/dev/null
done | awk -F"'" '
    /^ *name *:/ { name = $2; next }
    /^ *type *:/ && name != "" { print name, $2; name = "" }
' | sort -u > "$work/declared"

missing=()
for want in \
    "onAudio ([F)Z" \
    "nativeLoad (Ljava/lang/String;Ljava/lang/String;I)J" \
    nativeFree nativeSampleRate nativeLoadVoice nativeFreeVoice \
    nativeNewCancel nativeCancel nativeFreeCancel nativeSynthesize; do
    case "$want" in
        *" "*) grep -qxF "$want" "$work/declared" || missing+=("$want") ;;
        *) grep -q "^$want " "$work/declared" || missing+=("$want") ;;
    esac
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "FAIL: $(basename "$apk") does not declare:" >&2
    printf '      %s\n' "${missing[@]}" >&2
    echo "      The engine cannot reach them from native code, so this build" >&2
    echo "      fails on the first read. Check proguard-rules.pro." >&2
    exit 1
fi

echo "OK: $(basename "$apk") declares onAudio([F)Z and all nine native methods"
