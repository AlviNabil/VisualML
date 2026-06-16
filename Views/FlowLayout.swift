//
//  FlowLayout.swift
//  VisualML
//
//  A simple left-to-right "flow" / "wrap" layout: it places each child at its
//  natural size and wraps to the next line when the current line is full —
//  exactly how word-chips / tags should behave. Built on SwiftUI's `Layout`
//  protocol (iOS 16+), so it works inside any container.
//

import SwiftUI

struct FlowLayout: Layout {
    /// Gap between items both horizontally and vertically.
    var spacing: CGFloat = 8

    /// Tell the parent how tall we need to be for the proposed width.
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0          // running x within the current line
        var lineHeight: CGFloat = 0 // tallest item on the current line
        var totalHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)   // each chip's natural size
            // If this item won't fit on the current line, wrap to a new line.
            if x > 0, x + size.width > maxWidth {
                totalHeight += lineHeight + spacing
                x = 0
                lineHeight = 0
            }
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        totalHeight += lineHeight
        return CGSize(width: proposal.width ?? x, height: totalHeight)
    }

    /// Actually position each child.
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            // Wrap when the next chip would overflow the right edge.
            if x > bounds.minX, x + size.width > bounds.maxX {
                x = bounds.minX
                y += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), anchor: .topLeading,
                          proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
