import AppKit
import SwiftUI
import UserNotifications

class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    private var statusItem: NSStatusItem!
    private var menu: NSMenu!  // Strong reference to keep menu alive
    private var configManager: ConfigManager!
    private var voiceManager: VoiceManager!
    private var serverManager: ServerManager!

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("=== AppDelegate.applicationDidFinishLaunching ===")
        print("AppDelegate self: \(Unmanaged.passUnretained(self).toOpaque())")
        
        // Request notification permissions (only if running as proper app bundle)
        if Bundle.main.bundleIdentifier != nil {
            UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, error in
                if let error = error {
                    print("Failed to request notification permissions: \(error)")
                } else {
                    print("✓ Notification permissions: \(granted ? "granted" : "denied")")
                }
            }
        } else {
            print("⚠ Skipping notification setup (not running as app bundle)")
        }

        // Initialize managers on main thread
        print("Initializing managers...")
        configManager = ConfigManager.shared
        voiceManager = VoiceManager.shared
        serverManager = ServerManager.shared
        print("✓ Managers initialized")

        // Create menu bar item
        print("Creating status item...")
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        print("✓ Status item created: \(statusItem != nil)")
        print("  Status item address: \(Unmanaged.passUnretained(statusItem!).toOpaque())")

        if let button = statusItem.button {
            print("✓ Status item button exists")
            print("  Button address: \(Unmanaged.passUnretained(button).toOpaque())")
            
            // Use SF Symbol for microphone icon
            button.image = NSImage(systemSymbolName: "mic.fill", accessibilityDescription: "Pocket TTS")
            print("✓ Button image set: \(button.image != nil)")
            
            // Add click action for debugging
            button.action = #selector(statusItemClicked(_:))
            button.target = self
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
            print("✓ Button action configured for click debugging")
        } else {
            print("✗ ERROR: Status item button is nil!")
        }

        // Build and attach menu
        print("About to call updateMenu()...")
        updateMenu()
        
        // Verify menu is attached IMMEDIATELY after updateMenu()
        if let attachedMenu = statusItem.menu {
            print("✓ Menu confirmed attached with \(attachedMenu.items.count) items")
            print("  Menu address: \(Unmanaged.passUnretained(attachedMenu).toOpaque())")
            print("  self.menu address: \(Unmanaged.passUnretained(self.menu!).toOpaque())")
            print("  Same object: \(attachedMenu === self.menu)")
        } else {
            print("✗ ERROR: statusItem.menu is nil after updateMenu()!")
            print("  But self.menu is: \(self.menu != nil ? "not nil" : "nil")")
            if let m = self.menu {
                print("  self.menu has \(m.items.count) items")
            }
        }

        // Update menu when server status changes
        setupObservers()

        print("=== AppDelegate initialization complete ===")
        print("Voices loaded: \(voiceManager.voices.count)")
        print("Selected voice: \(configManager.config.selectedVoiceId)")
        
        // Schedule a check to verify everything is still connected after a delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            self?.verifyMenuBarState()
        }
    }
    
    @objc private func statusItemClicked(_ sender: Any?) {
        print("=== STATUS ITEM CLICKED ===")
        print("Timestamp: \(Date())")
        print("Sender: \(String(describing: sender))")
        print("Self still alive: \(true)") // If we get here, self is alive
        print("AppDelegate address: \(Unmanaged.passUnretained(self).toOpaque())")
        
        // Check status item state
        if let si = statusItem {
            print("✓ statusItem exists")
            print("  statusItem.menu: \(si.menu != nil ? "exists with \(si.menu!.items.count) items" : "NIL!")")
            print("  statusItem.button: \(si.button != nil ? "exists" : "NIL!")")
            
            if let m = si.menu {
                print("  Menu delegate: \(String(describing: m.delegate))")
                print("  Menu items:")
                for (i, item) in m.items.enumerated() {
                    print("    [\(i)] \(item.title) - enabled:\(item.isEnabled) action:\(String(describing: item.action))")
                }
            }
        } else {
            print("✗ statusItem is NIL!")
        }
        
        // Check our stored menu reference
        if let m = menu {
            print("✓ self.menu exists with \(m.items.count) items")
        } else {
            print("✗ self.menu is NIL!")
        }
        
        // Manually pop up the menu if it exists
        if let button = statusItem?.button, let menu = statusItem?.menu {
            print("Manually calling menu.popUp()...")
            menu.popUp(positioning: nil, at: NSPoint(x: 0, y: button.bounds.height), in: button)
        }
    }
    
    private func verifyMenuBarState() {
        print("=== DELAYED VERIFICATION (2 seconds after init) ===")
        print("AppDelegate still exists: true")
        print("statusItem: \(statusItem != nil ? "exists" : "NIL")")
        print("self.menu: \(menu != nil ? "exists" : "NIL")")
        if let si = statusItem {
            print("statusItem.menu: \(si.menu != nil ? "exists with \(si.menu!.items.count) items" : "NIL")")
            print("statusItem.button: \(si.button != nil ? "exists" : "NIL")")
        }
    }
    
    // MARK: - NSMenuDelegate
    
    func menuWillOpen(_ menu: NSMenu) {
        print("=== MENU WILL OPEN ===")
        print("Menu: \(Unmanaged.passUnretained(menu).toOpaque())")
        print("Items: \(menu.items.count)")
    }
    
    func menuDidClose(_ menu: NSMenu) {
        print("=== MENU DID CLOSE ===")
    }

    private func setupObservers() {
        // Observe server status changes to update menu
        Task { @MainActor in
            for await _ in NotificationCenter.default.notifications(named: NSNotification.Name("ServerStatusChanged")) {
                updateMenu()
            }
        }
    }

    @MainActor
    private func updateMenu() {
        print("=== updateMenu() called ===")
        print("VoiceManager has \(voiceManager.voices.count) voices")

        self.menu = NSMenu()
        self.menu.delegate = self  // Set delegate for open/close debugging
        print("Created new NSMenu at: \(Unmanaged.passUnretained(self.menu!).toOpaque())")

        // Header
        let headerItem = NSMenuItem(title: "Pocket TTS", action: nil, keyEquivalent: "")
        headerItem.isEnabled = false
        self.menu.addItem(headerItem)

        // Server status
        let statusText = serverStatusText()
        let statusMenuItem = NSMenuItem(title: statusText, action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        self.menu.addItem(statusMenuItem)

        self.menu.addItem(NSMenuItem.separator())

        // Select Voice submenu
        let voiceMenu = NSMenu()
        let selectedVoiceId = configManager.config.selectedVoiceId
        print("Selected voice ID: \(selectedVoiceId)")

        for voice in voiceManager.voices {
            let voiceItem = NSMenuItem(
                title: "\(voice.name) (\(voice.type.rawValue))",
                action: #selector(selectVoice(_:)),
                keyEquivalent: ""
            )
            voiceItem.target = self
            voiceItem.representedObject = voice

            // Add checkmark for selected voice
            if voice.id == selectedVoiceId {
                voiceItem.state = .on
                print("  ✓ \(voice.name) (SELECTED)")
            } else {
                print("    \(voice.name)")
            }

            voiceMenu.addItem(voiceItem)
        }
        print("Voice submenu has \(voiceMenu.items.count) items")

        let selectVoiceItem = NSMenuItem(title: "Select Voice", action: nil, keyEquivalent: "")
        selectVoiceItem.submenu = voiceMenu
        self.menu.addItem(selectVoiceItem)

        self.menu.addItem(NSMenuItem.separator())

        // Refresh Voices
        let refreshItem = NSMenuItem(
            title: "Refresh Voices",
            action: #selector(refreshVoices),
            keyEquivalent: "r"
        )
        refreshItem.target = self
        self.menu.addItem(refreshItem)

        // Check Server Status
        let checkServerItem = NSMenuItem(
            title: "Check Server Status",
            action: #selector(checkServerStatus),
            keyEquivalent: ""
        )
        checkServerItem.target = self
        self.menu.addItem(checkServerItem)

        self.menu.addItem(NSMenuItem.separator())

        // Open Main App
        let openAppItem = NSMenuItem(
            title: "Open Main App",
            action: #selector(openMainApp),
            keyEquivalent: ""
        )
        openAppItem.target = self
        self.menu.addItem(openAppItem)

        self.menu.addItem(NSMenuItem.separator())

        // Quit
        let quitItem = NSMenuItem(
            title: "Quit",
            action: #selector(quit),
            keyEquivalent: "q"
        )
        quitItem.target = self
        self.menu.addItem(quitItem)

        // CRITICAL: Assign menu to status item
        print("About to assign menu to statusItem...")
        print("  statusItem: \(statusItem != nil ? "exists" : "NIL!")")
        print("  self.menu: \(self.menu != nil ? "exists with \(self.menu!.items.count) items" : "NIL!")")
        
        statusItem.menu = self.menu
        
        print("After assignment:")
        print("  statusItem.menu: \(statusItem.menu != nil ? "exists with \(statusItem.menu!.items.count) items" : "NIL!")")
        print("  Same object check: \(statusItem.menu === self.menu)")
    }

    @MainActor
    private func serverStatusText() -> String {
        switch serverManager.status {
        case .running:
            return "Server: Running ✓"
        case .stopped:
            return "Server: Stopped ✗"
        case .unknown:
            return "Server: Checking..."
        }
    }

    @objc private func selectVoice(_ sender: NSMenuItem) {
        guard let voice = sender.representedObject as? Voice else { return }

        Task { @MainActor in
            configManager.updateSelectedVoice(id: voice.id, type: voice.type)
            updateMenu()
        }
    }

    @objc private func refreshVoices() {
        Task { @MainActor in
            voiceManager.reload()
            configManager.reload()
            updateMenu()

            // Show notification using modern UserNotifications framework (only if running as app bundle)
            if Bundle.main.bundleIdentifier != nil {
                let content = UNMutableNotificationContent()
                content.title = "Pocket TTS"
                content.body = "Voices refreshed"

                let request = UNNotificationRequest(
                    identifier: UUID().uuidString,
                    content: content,
                    trigger: nil
                )

                do {
                    try await UNUserNotificationCenter.current().add(request)
                    print("✓ Notification displayed: Voices refreshed")
                } catch {
                    print("Failed to show notification: \(error)")
                }
            } else {
                print("✓ Voices refreshed (notification skipped - not running as app bundle)")
            }
        }
    }

    @objc private func checkServerStatus() {
        Task { @MainActor in
            await serverManager.checkHealth()
            updateMenu()
        }
    }

    @objc private func openMainApp() {
        // Try to find and launch the Electron app
        let electronAppPaths = [
            "/Applications/Pocket TTS.app",
            "\(NSHomeDirectory())/Applications/Pocket TTS.app"
        ]

        for path in electronAppPaths {
            if FileManager.default.fileExists(atPath: path) {
                NSWorkspace.shared.openApplication(
                    at: URL(fileURLWithPath: path),
                    configuration: NSWorkspace.OpenConfiguration()
                ) { _, error in
                    if let error = error {
                        print("Failed to open main app: \(error)")
                        self.showErrorAlert("Could not open main app: \(error.localizedDescription)")
                    }
                }
                return
            }
        }

        showErrorAlert("Pocket TTS main app not found. Please install it first.")
    }

    @objc private func quit() {
        NSApplication.shared.terminate(nil)
    }

    private func showErrorAlert(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Pocket TTS"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
    
    deinit {
        print("⚠️⚠️⚠️ AppDelegate DEINITIALIZED! This should NOT happen while app is running! ⚠️⚠️⚠️")
    }
}
