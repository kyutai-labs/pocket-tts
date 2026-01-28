import AppKit

@main
struct PocketTTSMenuBarApp {
    // CRITICAL: Keep a strong reference to prevent ARC from deallocating the delegate!
    // NSApplication.delegate is a WEAK reference, so without this, the delegate gets
    // deallocated immediately after app.run() starts the run loop.
    static var appDelegate: AppDelegate!
    
    static func main() {
        print("=== PocketTTSMenuBarApp.main() starting ===")
        
        let app = NSApplication.shared
        print("NSApplication.shared obtained")
        
        // Create delegate and store in STATIC variable (strong reference)
        appDelegate = AppDelegate()
        print("AppDelegate created and stored in static var")
        
        app.delegate = appDelegate
        print("AppDelegate assigned to app.delegate (weak ref)")
        print("appDelegate still alive: \(appDelegate != nil)")
        
        app.setActivationPolicy(.accessory)
        print("Activation policy set to .accessory (menu bar only, no dock icon)")
        
        print("=== About to call app.run() - entering run loop ===")
        app.run()
        
        // This line will never execute (run loop blocks until termination)
        print("app.run() returned (app terminating)")
    }
}
