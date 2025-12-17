import Cocoa
import Foundation

// Global state
var cursorPoint: NSPoint = NSPoint(x: 0, y: 0)
var showOverlay: Bool = true

class OverlayView: NSView {
    override func draw(_ dirtyRect: NSRect) {
        // Clear background
        NSColor.clear.set()
        dirtyRect.fill()
        
        let screenW = self.bounds.width
        let screenH = self.bounds.height
        
        // 1. Draw Grid (Faint White)
        let gridColor = NSColor(white: 1.0, alpha: 0.1)
        gridColor.setStroke()
        let path = NSBezierPath()
        path.lineWidth = 1.0
        
        // Verticals
        for x in stride(from: 0.0, to: screenW, by: 100.0) {
            path.move(to: NSPoint(x: x, y: 0))
            path.line(to: NSPoint(x: x, y: screenH))
        }
        // Horizontals
        for y in stride(from: 0.0, to: screenH, by: 100.0) {
            path.move(to: NSPoint(x: 0, y: y))
            path.line(to: NSPoint(x: screenW, y: y))
        }
        path.stroke()
        
        // 2. Draw Cursor
        // Invert Y because Cocoa is bottom-up vs our Top-Down logic
        // We will assume Rust sends Top-Down coordinates (0,0 is Top-Left).
        // So Cocoa Y = Height - InputY
        let drawY = screenH - cursorPoint.y
        let drawX = cursorPoint.x
        
        // Large Circle (Blue, Alpha 0.5)
        let lRadius: CGFloat = 50.0
        let lRect = NSRect(x: drawX - lRadius, y: drawY - lRadius, width: lRadius * 2, height: lRadius * 2)
        NSColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 0.4).setFill()
        NSBezierPath(ovalIn: lRect).fill()
        
        // Center Dot (Red, Opaque)
        let sRadius: CGFloat = 5.0
        let sRect = NSRect(x: drawX - sRadius, y: drawY - sRadius, width: sRadius * 2, height: sRadius * 2)
        NSColor.red.setFill()
        NSBezierPath(ovalIn: sRect).fill()
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        let screenRect = NSScreen.main?.frame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        
        window = NSWindow(
            contentRect: screenRect,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        
        window.level = .floating // or NSWindow.Level(rawValue: 25)
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = false
        window.ignoresMouseEvents = true
        
        let view = OverlayView(frame: screenRect)
        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        
        // Start Stdin Reader
        DispatchQueue.global(qos: .userInteractive).async {
            while let line = readLine(strippingNewline: true) {
                let parts = line.split(separator: " ")
                if parts.count >= 2, 
                   let x = Double(parts[0]), 
                   let y = Double(parts[1]) {
                    
                    DispatchQueue.main.async {
                        cursorPoint = NSPoint(x: x, y: y)
                        view.needsDisplay = true
                    }
                }
            }
            // If stdin closes, exit
            DispatchQueue.main.async {
                NSApplication.shared.terminate(nil)
            }
        }
    }
}

// Main Entry
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
