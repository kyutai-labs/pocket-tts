plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "org.pockettts.android"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.pockettts.android"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // Phones shipping today are all arm64; armeabi-v7a for older hardware,
        // x86_64 for the emulator. Each ABI costs its own ONNX Runtime (18 to
        // 31 MB), so the list is what the app is run on and nothing more.
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    // The debug key is committed, and this is the one decision here that looks
    // wrong until you have been bitten by the alternative.
    //
    // Left to itself the Android plugin signs debug builds with
    // ~/.android/debug.keystore, generating one if it is missing. A CI runner
    // is a fresh machine every time, so every run generated a *new* key pair
    // and every APK this project has ever published was signed by a different
    // identity. Android identifies an app by its signature, so the consequences
    // were permanent rather than occasional: no build could ever be installed
    // over another - INSTALL_FAILED_UPDATE_INCOMPATIBLE, every time - and
    // keeping app data on uninstall, which hasFragileUserData exists to offer,
    // made the *next* install fail outright, because retained data remembers
    // the identity that wrote it. The way through was to uninstall and discard
    // the data, which costs a 98 MB model download on every single sideload.
    // That is the tax this removes.
    //
    // What committing it gives away: anyone with this repository can build an
    // APK that installs over yours. That matters for an app shipped through a
    // store and it is why a release key is never committed. This key is not
    // that - it signs a debug key'd build of an app distributed by handing
    // someone a file - and the same trade is already made by the platform's
    // own debug key, which is public and identical on every machine.
    signingConfigs {
        getByName("debug") {
            storeFile = file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            // Signed with the debug key so the shrunk build can actually be
            // installed and tested. R8 is the part of the build no test sees,
            // and it has already removed a JNI callback once; an artefact
            // nobody can sideload is an artefact nobody checks. This is not a
            // distribution key and this app is not distributed through a store.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    testOptions {
        unitTests {
            isReturnDefaultValues = true
            // Robolectric boots the real activities against real resources, so
            // a theme or manifest mistake fails the build instead of the phone.
            isIncludeAndroidResources = true
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.preference.ktx)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.recyclerview)
    implementation(libs.material)
    implementation(libs.kotlinx.coroutines.android)

    // ONNX Runtime for pocket-speak's engine, which opens libonnxruntime.so
    // by name (pocket-speak/crates/android). 1.24.2 is the version the desk CLI runs,
    // which is the version its measurements were taken on. Its Java API comes
    // along and is unused.
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.androidx.test.junit)
}

// pocket-speak's engine, built from ../pocket-speak into src/main/jniLibs for every
// ABI above. Needs rustup's Android targets, cargo-ndk and the NDK; set
// -PskipNative to package whatever is already there.
val buildNative by tasks.registering(Exec::class) {
    val abis = listOf("arm64-v8a", "armeabi-v7a", "x86_64")
    workingDir = rootProject.file("../pocket-speak")
    commandLine(
        listOf("cargo", "ndk", "--platform", "26", "-o", file("src/main/jniLibs").absolutePath) +
            abis.flatMap { listOf("-t", it) } +
            listOf("build", "--release", "-p", "pocket-tts-android", "--lib"),
    )
    onlyIf { !project.hasProperty("skipNative") }
}
tasks.named("preBuild") { dependsOn(buildNative) }
