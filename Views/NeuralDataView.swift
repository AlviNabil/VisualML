//
//  NeuralDataView.swift
//  VisualML
//
//  Step 1 of the neural network walkthrough: a dataset where every model built
//  so far in this app fails, because all of them draw a straight boundary and
//  the boundary needed here is a closed curve.
//

import SwiftUI

struct NeuralDataView: View {
    let export: NeuralExport

    @State private var showBaseline = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                NeuralScatterView(export: export, showStraightLine: showBaseline)
                    .frame(height: 300)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                legend
                Toggle("Try the best straight boundary", isOn: $showBaseline.animation())
                    .font(.subheadline)

                if showBaseline { baselineVerdict }
                whatIsNeeded
            }
            .padding()
        }
        .navigationTitle("1 · The data")
        .navigationBarTitleDisplayMode(.inline)
        .neuralInfo(export, topic: .setup)
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("A boundary that curves").font(.headline)
            Text("One class sits in the middle; the other forms a ring around "
                 + "it. The two never overlap — but the line between them closes "
                 + "on itself, and nothing built so far in this app can draw a "
                 + "shape like that.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var legend: some View {
        HStack(spacing: 16) {
            Label { Text(export.dataset.classNames[0]).font(.caption) } icon: {
                Circle().fill(.blue).frame(width: 9, height: 9)
            }
            Label { Text(export.dataset.classNames[1]).font(.caption) } icon: {
                Circle().fill(.orange).frame(width: 9, height: 9)
            }
            Spacer()
            Text("\(export.dataset.n) points").font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var baselineVerdict: some View {
        let accuracy = export.dataset.linearBaselineAccuracy
        return VStack(alignment: .leading, spacing: 8) {
            Text("Barely better than guessing").font(.headline)
            HStack(spacing: 10) {
                stat("straight boundary", String(format: "%.1f%%", accuracy * 100), .red)
                stat("coin flip", "50.0%", .secondary)
                stat("this network", String(format: "%.1f%%",
                                            export.training.finalAccuracy * 100), .green)
            }
            Text("Logistic regression, an SVM, the LSA classifier — every model "
                 + "in this app before now separates classes with one straight "
                 + "cut. Whichever way that cut is angled, it leaves roughly half "
                 + "the ring on the wrong side.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.red.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var whatIsNeeded: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("What would work").font(.headline)
            Text("A model that can bend its boundary — ideally into a closed "
                 + "loop around the middle class. A neural network builds "
                 + "exactly that by stacking several straight cuts and then "
                 + "bending the result, which is what the next steps take apart.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func stat(_ title: String, _ value: String, _ tint: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit()).foregroundStyle(tint)
            Text(title).font(.caption2).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - The scatter

/// The two classes on the plane, optionally over a probability map from the
/// network, and optionally with a straight boundary drawn to show it failing.
struct NeuralScatterView: View {
    let export: NeuralExport
    /// A probability grid from the network, drawn underneath the points.
    var grid: [[Double]]? = nil
    var showStraightLine: Bool = false
    /// Draw only the boundary contour rather than a filled map.
    var contourOnly: Bool = false

    var body: some View {
        Canvas { context, size in
            let lo = export.training.gridMin, hi = export.training.gridMax
            let pad: CGFloat = 14
            let side = Swift.min(size.width, size.height) - 2 * pad
            let originX = (size.width - side) / 2
            let originY = (size.height - side) / 2

            func sx(_ v: Double) -> CGFloat {
                originX + CGFloat((v - lo) / (hi - lo)) * side
            }
            func sy(_ v: Double) -> CGFloat {
                originY + side - CGFloat((v - lo) / (hi - lo)) * side
            }

            // The network's probability map, if one was supplied.
            if let grid, !contourOnly {
                let steps = grid.count
                let cell = side / CGFloat(steps - 1)
                for (row, values) in grid.enumerated() {
                    for (column, p) in values.enumerated() {
                        let x = lo + (hi - lo) * Double(column) / Double(steps - 1)
                        let y = lo + (hi - lo) * Double(row) / Double(steps - 1)
                        let rect = CGRect(x: sx(x) - cell / 2, y: sy(y) - cell / 2,
                                          width: cell + 1, height: cell + 1)
                        // Blue where the network says class 0, orange for class 1.
                        let tilt = p - 0.5
                        let colour = (tilt >= 0 ? Color.orange : Color.blue)
                            .opacity(0.10 + 0.55 * Swift.min(abs(tilt) * 2, 1))
                        context.fill(Path(rect), with: .color(colour))
                    }
                }
            }

            // The 0.5 contour, traced by marching along grid cells.
            if let grid {
                let steps = grid.count
                var boundary = Path()
                for row in 0..<(steps - 1) {
                    for column in 0..<(steps - 1) {
                        let corners = [grid[row][column], grid[row][column + 1],
                                       grid[row + 1][column + 1], grid[row + 1][column]]
                        let above = corners.filter { $0 >= 0.5 }.count
                        guard above > 0, above < 4 else { continue }
                        let x = lo + (hi - lo) * Double(column) / Double(steps - 1)
                        let y = lo + (hi - lo) * Double(row) / Double(steps - 1)
                        let cell = side / CGFloat(steps - 1)
                        boundary.addEllipse(in: CGRect(x: sx(x) - cell / 2,
                                                       y: sy(y) - cell / 2,
                                                       width: cell, height: cell))
                    }
                }
                context.fill(boundary, with: .color(.primary.opacity(0.55)))
            }

            // A straight boundary, angled to show that no orientation helps.
            if showStraightLine {
                var line = Path()
                line.move(to: CGPoint(x: sx(lo), y: sy(lo + (hi - lo) * 0.30)))
                line.addLine(to: CGPoint(x: sx(hi), y: sy(lo + (hi - lo) * 0.70)))
                context.stroke(line, with: .color(.red),
                               style: StrokeStyle(lineWidth: 2.5, dash: [6, 4]))
            }

            // The points.
            for point in export.dataset.points {
                let centre = CGPoint(x: sx(point.x[0]), y: sy(point.x[1]))
                let radius: CGFloat = 3
                let colour: Color = point.label == 0 ? .blue : .orange
                context.fill(Path(ellipseIn: CGRect(x: centre.x - radius,
                                                    y: centre.y - radius,
                                                    width: 2 * radius, height: 2 * radius)),
                             with: .color(colour.opacity(0.85)))
            }
        }
    }
}
