//
//  NetworkDiagramView.swift
//  VisualML
//
//  The network drawn as neurons and connections. Shared by the architecture,
//  forward-pass and backpropagation steps: the same picture each time, with
//  different quantities mapped onto the node fills and the edge colours, so
//  the shape stays familiar while the story changes.
//

import SwiftUI

/// What a value means when it is painted onto a node or an edge.
enum SignalScale {
    /// Activations in [-1, 1] (tanh) or [0, 1] (sigmoid, ReLU): signed colour.
    case signed
    /// Magnitudes where only size matters, drawn as opacity on one hue.
    case magnitude(Color)
}

struct NetworkDiagramView: View {
    let layerSizes: [Int]

    /// Per-layer node values, including the input layer, so
    /// `nodeValues[0]` is the input and `nodeValues.last` the output.
    var nodeValues: [[Double]]? = nil
    /// Per-layer weight matrices, shaped (fanIn, fanOut).
    var weights: [[[Double]]]? = nil
    /// How to colour the node values.
    var nodeScale: SignalScale = .signed
    /// Draw the arrows pointing backward, for the backpropagation step.
    var reversed: Bool = false
    /// Layer index (0-based, counting the input) to emphasise; others dim.
    var focusLayer: Int? = nil
    var showValues: Bool = true

    var body: some View {
        Canvas { context, size in
            let padX: CGFloat = 30
            let padY: CGFloat = 22
            let columns = layerSizes.count
            guard columns > 1 else { return }

            let stepX = (size.width - 2 * padX) / CGFloat(columns - 1)

            func centre(_ layer: Int, _ unit: Int) -> CGPoint {
                let count = layerSizes[layer]
                let usable = size.height - 2 * padY
                let spacing = count > 1 ? usable / CGFloat(count - 1) : 0
                let y = count > 1
                    ? padY + CGFloat(unit) * spacing
                    : size.height / 2
                return CGPoint(x: padX + CGFloat(layer) * stepX, y: y)
            }

            // Connections first, so nodes sit on top of them.
            if let weights {
                for (index, matrix) in weights.enumerated() {
                    let dimmed = focusLayer != nil && focusLayer != index + 1
                    for (i, row) in matrix.enumerated() {
                        for (j, w) in row.enumerated() {
                            let from = centre(index, i)
                            let to = centre(index + 1, j)
                            var path = Path()
                            path.move(to: from)
                            path.addLine(to: to)
                            let strength = Swift.min(abs(w) / 2.0, 1.0)
                            let colour: Color = w >= 0 ? .blue : .red
                            let alpha = dimmed ? 0.06 : (0.15 + 0.6 * strength)
                            context.stroke(path, with: .color(colour.opacity(alpha)),
                                           lineWidth: dimmed ? 0.5 : (0.6 + 2.0 * strength))
                        }
                    }
                }
            } else {
                // No weights supplied: plain grey wiring.
                for index in 0..<(columns - 1) {
                    for i in 0..<layerSizes[index] {
                        for j in 0..<layerSizes[index + 1] {
                            var path = Path()
                            path.move(to: centre(index, i))
                            path.addLine(to: centre(index + 1, j))
                            context.stroke(path, with: .color(.gray.opacity(0.28)),
                                           lineWidth: 0.8)
                        }
                    }
                }
            }

            // Direction arrows along the top.
            let arrowY = size.height - 6
            var arrow = Path()
            let startX = reversed ? size.width - padX : padX
            let endX = reversed ? padX : size.width - padX
            arrow.move(to: CGPoint(x: startX, y: arrowY))
            arrow.addLine(to: CGPoint(x: endX, y: arrowY))
            context.stroke(arrow, with: .color(.secondary.opacity(0.5)),
                           style: StrokeStyle(lineWidth: 1, dash: [4, 3]))

            // Nodes.
            for layer in 0..<columns {
                let dimmed = focusLayer != nil && focusLayer != layer
                for unit in 0..<layerSizes[layer] {
                    let point = centre(layer, unit)
                    let radius: CGFloat = 13
                    let rect = CGRect(x: point.x - radius, y: point.y - radius,
                                      width: 2 * radius, height: 2 * radius)

                    var fill = Color.gray.opacity(0.25)
                    if let nodeValues, layer < nodeValues.count,
                       unit < nodeValues[layer].count {
                        let value = nodeValues[layer][unit]
                        switch nodeScale {
                        case .signed:
                            let magnitude = Swift.min(abs(value), 1.0)
                            fill = (value >= 0 ? Color.blue : Color.orange)
                                .opacity(0.15 + 0.75 * magnitude)
                        case .magnitude(let hue):
                            let magnitude = Swift.min(abs(value) * 4, 1.0)
                            fill = hue.opacity(0.12 + 0.8 * magnitude)
                        }
                    }
                    if dimmed { fill = fill.opacity(0.25) }

                    context.fill(Path(ellipseIn: rect), with: .color(fill))
                    context.stroke(Path(ellipseIn: rect),
                                   with: .color(.primary.opacity(dimmed ? 0.15 : 0.45)),
                                   lineWidth: 1)

                    if showValues, let nodeValues, layer < nodeValues.count,
                       unit < nodeValues[layer].count, !dimmed {
                        let text = String(format: "%.2f", nodeValues[layer][unit])
                        context.draw(Text(text).font(.system(size: 8, design: .monospaced))
                                        .foregroundStyle(.primary),
                                     at: point)
                    }
                }
            }
        }
    }
}

/// A compact legend explaining the colours the diagram uses for weights.
struct WeightLegend: View {
    var body: some View {
        HStack(spacing: 14) {
            label("positive weight", .blue)
            label("negative weight", .red)
            Text("thicker = larger")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func label(_ text: String, _ colour: Color) -> some View {
        HStack(spacing: 5) {
            Rectangle().fill(colour.opacity(0.7)).frame(width: 14, height: 2.5)
            Text(text).font(.caption2).foregroundStyle(.secondary)
        }
    }
}
