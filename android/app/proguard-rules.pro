# pocket-speak's engine calls back into these from JNI by name: the native
# methods themselves, and NativeEngine.Sink.onAudio(float[]) on whatever
# implements it ("([F)Z" in pocket-speak/crates/android). R8 would otherwise rename or
# inline them, and a JNI lookup that fails leaves a pending exception the
# engine turns into "audio sink" errors - a release build reading nothing.
# Kept outright, not just named: R8 otherwise drops a native method nothing
# calls in some build, and check-jni-callback.sh holds all nine.
-keepclasseswithmembers class org.pockettts.android.engine.NativeEngine {
    native <methods>;
}
-keep interface org.pockettts.android.engine.NativeEngine$Sink { *; }
-keepclassmembers class * implements org.pockettts.android.engine.NativeEngine$Sink {
    boolean onAudio(float[]);
}
