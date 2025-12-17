import Cocoa
import Foundation

// Global state
var cursorPoint: NSPoint = NSPoint(x: 0, y: 0)
var moondreamPoint: NSPoint? = nil
var capturedPoint: NSPoint? = nil
var capturedState: Int = -1 // 0=Pending(Red), 1=Done(Yellow), -1=Default(White)
var showOverlay: Bool = true
var customFontName: String = "Monospace"
var customFontSize: CGFloat = 14.0
var menuState: String = ""

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
        
        // 2. Real-time Gaze (Blue/Red - Flying)
        // Invert Y
        let gazeY = screenH - cursorPoint.y
        let gazeX = cursorPoint.x
        
        // Large Circle (Blue, Alpha 0.5)
        let lRadius: CGFloat = 50.0
        let lRect = NSRect(x: gazeX - lRadius, y: gazeY - lRadius, width: lRadius * 2, height: lRadius * 2)
        NSColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 0.4).setFill()
        NSBezierPath(ovalIn: lRect).fill()
        
        // Center Dot (Red, Opaque)
        let sRadius: CGFloat = 5.0
        let sRect = NSRect(x: gazeX - sRadius, y: gazeY - sRadius, width: sRadius * 2, height: sRadius * 2)
        NSColor.red.setFill()
        NSBezierPath(ovalIn: sRect).fill()
        
        // 3. Captured ONNX Gaze (Green/White) - "Where ONNX thought we looked at frame time"
        if let cap = capturedPoint {
            let capY = screenH - cap.y
            let capX = cap.x
            
            let cRadius: CGFloat = 40.0
            let cRect = NSRect(x: capX - cRadius, y: capY - cRadius, width: cRadius * 2, height: cRadius * 2)
            NSColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 0.5).setFill() // Green
            NSBezierPath(ovalIn: cRect).fill()
            
            let csRadius: CGFloat = 5.0
            let csRect = NSRect(x: capX - csRadius, y: capY - csRadius, width: csRadius * 2, height: csRadius * 2)
            
            // Color based on State
            if capturedState == 0 {
                NSColor.red.setFill() // Pending
            } else if capturedState == 1 {
                NSColor.yellow.setFill() // Done
            } else {
                NSColor.white.setFill() // Default
            }
            NSBezierPath(ovalIn: csRect).fill()
        }

        // 4. Moondream Gaze (Cyan/Gold)
        if let md = moondreamPoint {
            let mdY = screenH - md.y
            let mdX = md.x
            
            // Large Circle (Cyan, Alpha 0.6)
            let mRadius: CGFloat = 40.0
            let mRect = NSRect(x: mdX - mRadius, y: mdY - mRadius, width: mRadius * 2, height: mRadius * 2)
            NSColor(red: 0.0, green: 1.0, blue: 1.0, alpha: 0.6).setFill() // Cyan
            NSBezierPath(ovalIn: mRect).fill()
            
            // Center Dot (Gold)
            let msRadius: CGFloat = 6.0
            let msRect = NSRect(x: mdX - msRadius, y: mdY - msRadius, width: msRadius * 2, height: msRadius * 2)
            NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0).setFill()
            NSBezierPath(ovalIn: msRect).fill()
        }
        
        // 5. HUD Text (Top-Left Only)
        let font: NSFont
        if customFontName == "Monospace" {
            font = NSFont.monospacedSystemFont(ofSize: customFontSize, weight: .bold)
        } else {
             font = NSFont(name: customFontName, size: customFontSize) ?? NSFont.monospacedSystemFont(ofSize: customFontSize, weight: .bold)
        }
        let attrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.white,
            .strokeColor: NSColor.black,
            .strokeWidth: -2.0
        ]
        
        // 5. HUD Stats (Restored 4 Corners + Center)
        let info = [
            String(format: "REALTIME:  %04.0f, %04.0f", cursorPoint.x, cursorPoint.y),
            capturedPoint != nil ? String(format: "CAPTURED:  %04.0f, %04.0f", capturedPoint!.x, capturedPoint!.y) : "CAPTURED:  ----, ----",
            moondreamPoint != nil ? String(format: "MOONDREAM: %04.0f, %04.0f", moondreamPoint!.x, moondreamPoint!.y) : "MOONDREAM: ----, ----"
        ].joined(separator: "\n")
        
        let offsets = [
            NSPoint(x: 20, y: screenH - 120), // TL
            NSPoint(x: screenW - 250, y: screenH - 120), // TR (Moved right)
            NSPoint(x: 20, y: 40), // BL
            NSPoint(x: screenW - 250, y: 40), // BR (Moved right)
            NSPoint(x: screenW/2 - 150, y: screenH/2 - 40) // Center
        ]
        
        for p in offsets {
            info.draw(at: p, withAttributes: attrs)
        }
        
        // 6. Menu (Vertically Centered, Left Aligned)
        if !menuState.isEmpty {
            let menuHeight = CGFloat(menuState.split(separator: "\n").count) * (customFontSize + 4)
            // Center vertically: (ScreenH - Height) / 2
            let menuY = (screenH - menuHeight) / 2
            menuState.draw(at: NSPoint(x: 20, y: menuY), withAttributes: attrs)
        }
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
                
                // Protocol: "G x y" or "M x y"
                // Legacy support "x y" -> G
                if parts.count >= 2 {
                    if parts[0] == "G" && parts.count >= 3 {
                         if let x = Double(parts[1]), let y = Double(parts[2]) {
                             DispatchQueue.main.async {
                                 cursorPoint = NSPoint(x: x, y: y)
                                 view.needsDisplay = true
                             }
                         }
                    } else if parts[0] == "M" && parts.count >= 3 {
                         if let x = Double(parts[1]), let y = Double(parts[2]) {
                             DispatchQueue.main.async {
                                 moondreamPoint = NSPoint(x: x, y: y)
                                 view.needsDisplay = true
                             }
                         }
                    } else if parts[0] == "C" && parts.count >= 3 {
                         if let x = Double(parts[1]), let y = Double(parts[2]) {
                             let state = (parts.count >= 4) ? (Int(parts[3]) ?? -1) : -1
                             DispatchQueue.main.async {
                                 capturedPoint = NSPoint(x: x, y: y)
                                 capturedState = state
                                 view.needsDisplay = true
                             }
                         }
                    } else if parts[0] == "F" && parts.count >= 3 {
                        // Protocol: F <FamilyName> <Size>
                        // Note: Family Name might have spaces? Protocol needs to handle that.
                        // For now, assume single word or simple join.
                        
                        let size = Double(parts.last!) ?? 14.0
                        let nameParts = parts[1..<parts.count-1]
                        let name = nameParts.joined(separator: " ")
                        
                        DispatchQueue.main.async {
                            // Update Global Font State
                            customFontName = String(name)
                            customFontSize = CGFloat(size)
                            view.needsDisplay = true
                        }
                    } else if parts[0] == "S" && parts.count >= 2 {
                        // Protocol: S <Line1>| <Line2>| ...
                        // Used for Menu State
                        let raw = parts[1..<parts.count].joined(separator: " ")
                        let formatted = raw.replacingOccurrences(of: "|", with: "\n")
                        
                        DispatchQueue.main.async {
                            menuState = formatted
                            view.needsDisplay = true
                        }
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
