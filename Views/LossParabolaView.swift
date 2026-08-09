//
//  LossParabolaView.swift
//  VisualML
//
//  Shows what one gradient-descent step is actually doing. Holding every other
//  parameter fixed and sweeping a single one traces a parabola — a slice through
//  the loss surface. The current parameters sit somewhere on that curve, the
//  tangent shows the slope the gradient measures, and the arrow shows where the
//  next step lands.
//

import SwiftUI

/// Draws the loss slice along one parameter, with the tangent at the current
/// value and the step gradient descent is about to take.
struct LossParabolaView: View {
    let quadratic: LossQuadratic
    let theta: [Double]
    let optimum: [Double]
    let learningRate: Double
    /// Which entry of `theta` to sweep.
    let axis: Int

    var body: some View {
        Canvas { context, size in
            let current = theta[axis]
            let bottom = quadratic.coordinateMinimum(axis, theta)
            let slope = quadratic.gradient(theta)[axis]
            let next = current - learningRate * slope

            // Sweep a window that always contains the current point, the bottom
            // of the parabola and the point the next step reaches.
            let interesting = [current, bottom, next, optimum[axis]]
            let span = max((interesting.max()! - interesting.min()!) * 0.75,
                           abs(bottom) * 0.35 + 0.5)
            let lo = min(current, bottom, next) - span
            let hi = max(current, bottom, next) + span

            let samples = 120
            var values: [(t: Double, j: Double)] = []
            for i in 0...samples {
                let t = lo + (hi - lo) * Double(i) / Double(samples)
                values.append((t, quadratic.loss(theta, varying: axis, to: t)))
            }

            let pad: CGFloat = 20
            let jLo = values.map(\.j).min()!
            let jHi = values.map(\.j).max()!
            func sx(_ t: Double) -> CGFloat {
                pad + CGFloat((t - lo) / max(hi - lo, 1e-12)) * (size.width - 2 * pad)
            }
            func sy(_ j: Double) -> CGFloat {
                size.height - pad
                    - CGFloat((j - jLo) / max(jHi - jLo, 1e-12)) * (size.height - 2 * pad)
            }

            // The parabola.
            var curve = Path()
            for (i, v) in values.enumerated() {
                let p = CGPoint(x: sx(v.t), y: sy(v.j))
                if i == 0 { curve.move(to: p) } else { curve.addLine(to: p) }
            }
            context.stroke(curve, with: .color(.primary.opacity(0.55)), lineWidth: 2)

            // The bottom of this slice.
            let bottomJ = quadratic.loss(theta, varying: axis, to: bottom)
            var floorMark = Path()
            floorMark.move(to: CGPoint(x: sx(bottom), y: sy(bottomJ)))
            floorMark.addLine(to: CGPoint(x: sx(bottom), y: size.height - pad))
            context.stroke(floorMark, with: .color(.gray.opacity(0.55)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))

            // The tangent: slope = dJ/dtheta at the current value.
            let currentJ = quadratic.loss(theta, varying: axis, to: current)
            let reach = (hi - lo) * 0.22
            var tangent = Path()
            tangent.move(to: CGPoint(x: sx(current - reach),
                                     y: sy(currentJ - slope * reach)))
            tangent.addLine(to: CGPoint(x: sx(current + reach),
                                        y: sy(currentJ + slope * reach)))
            context.stroke(tangent, with: .color(.orange), lineWidth: 2)

            // The step downhill, drawn along the curve.
            let nextJ = quadratic.loss(theta, varying: axis, to: next)
            var step = Path()
            step.move(to: CGPoint(x: sx(current), y: sy(currentJ)))
            step.addLine(to: CGPoint(x: sx(next), y: sy(nextJ)))
            context.stroke(step, with: .color(.accentColor),
                           style: StrokeStyle(lineWidth: 2, dash: [5, 3]))

            // Where we are now, and where the step lands.
            func dot(_ x: Double, _ j: Double, _ color: Color, _ r: CGFloat) {
                let c = CGPoint(x: sx(x), y: sy(j))
                context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r,
                                                    width: 2 * r, height: 2 * r)),
                             with: .color(color))
            }
            dot(next, nextJ, .accentColor.opacity(0.7), 4)
            dot(current, currentJ, .orange, 5.5)

            context.draw(Text("loss").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: 6, y: 8), anchor: .topLeading)
            context.draw(Text("parameter").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width - 6, y: size.height - 6),
                         anchor: .bottomTrailing)
        }
    }
}

/// The parabola plus a plain-language reading of the numbers behind it.
struct GradientStepExplainer: View {
    let export: RegressionExport
    let frame: RegressionFrame
    let learningRate: Double
    @State private var axis = 0

    var body: some View {
        let theta = frame.theta
        let slope = export.lossQuadratic.gradient(theta)[axis]
        let current = theta[axis]
        let next = current - learningRate * slope
        let names = parameterNames

        return VStack(alignment: .leading, spacing: 10) {
            Text("One step, on the loss curve").font(.headline)
            Text("Sweeping \(names[axis]) with the others held fixed traces this "
                 + "parabola. The orange line is the tangent — its steepness is the "
                 + "gradient. The step moves downhill, against that slope.")
                .font(.caption).foregroundStyle(.secondary)

            if names.count > 1 {
                Picker("Parameter", selection: $axis) {
                    ForEach(names.indices, id: \.self) { Text(names[$0]).tag($0) }
                }
                .pickerStyle(.segmented)
            }

            LossParabolaView(quadratic: export.lossQuadratic,
                             theta: theta,
                             optimum: export.closedForm.theta,
                             learningRate: learningRate,
                             axis: axis)
                .frame(height: 190)
                .padding(8)
                .background(Color.gray.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 3) {
                arithmetic("slope  ∂J/∂\(names[axis])", String(format: "%+.4f", slope))
                arithmetic("step  −η × slope",
                           String(format: "−%.6f × %+.4f = %+.4f",
                                  learningRate, slope, -learningRate * slope))
                arithmetic("\(names[axis])",
                           String(format: "%.4f → %.4f", current, next))
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            Text(slope > 0
                 ? "The slope is positive — the curve rises to the right, so the step goes left."
                 : "The slope is negative — the curve rises to the left, so the step goes right.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var parameterNames: [String] {
        export.dataset.featureNames.map { name in
            "w(\(name.replacingOccurrences(of: "_", with: " ")))"
        } + ["b"]
    }

    private func arithmetic(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospaced())
        }
    }
}
