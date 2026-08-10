//
//  LogisticTrainingView.swift
//  VisualML
//
//  Step 3 of the logistic walkthrough: finding the curve. The S starts flat and
//  is pushed into place by gradient descent, which can be watched either on the
//  data itself or as a path across the cross-entropy landscape.
//

import SwiftUI
import Combine

struct LogisticTrainingView: View {
    let export: LogisticExport

    @State private var runIndex = 2          // the rate that converges cleanly
    @State private var frame: Double = 0
    @State private var playing = false

    private let timer = Timer.publish(every: 0.06, on: .main, in: .common).autoconnect()

    var body: some View {
        let run = export.runs[runIndex]
        let last = max(run.history.count - 1, 0)
        let step = run.history[min(Int(frame), last)]

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                FittingCurveView(export: export, current: step)
                    .frame(height: 230)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                parameters(step)
                learningRatePicker
                playback(run: run, step: step, last: last)

                section("The loss as it falls") {
                    LogLossCurveView(history: run.history,
                                     best: export.fit.logLoss,
                                     currentIteration: step.iter)
                        .frame(height: 120)
                }

                section("The same descent, seen from above") {
                    VStack(alignment: .leading, spacing: 8) {
                        LossLandscapeView(surface: export.lossSurface,
                                          history: run.history,
                                          upTo: min(Int(frame), last),
                                          target: export.fit.theta)
                            .frame(height: 230)
                        Text("Every point is one pair of parameters, shaded by its "
                             + "cross-entropy — dark is better. The path is the run "
                             + "so far; the star is the best fit.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }

                scores(step)
            }
            .padding()
        }
        .navigationTitle("3 · Finding the curve")
        .navigationBarTitleDisplayMode(.inline)
        .onReceive(timer) { _ in
            guard playing else { return }
            if Int(frame) >= last { playing = false }
            else { frame = min(frame + 1, Double(last)) }
        }
        .onChange(of: runIndex) { _, _ in frame = 0; playing = false }
    }

    // MARK: - Copy

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Nudging the curve into place").font(.headline)
            Text("Training starts with w and b at zero, which makes σ return 0.5 "
                 + "for everyone — a flat line of pure indecision. Each step "
                 + "steepens and shifts it to match the outcomes.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func parameters(_ step: LogisticFrame) -> some View {
        let boundaryText = step.boundary.map { String(format: "%.2f h", $0) } ?? "—"
        return VStack(alignment: .leading, spacing: 4) {
            Text("Now")
                .font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "p = σ(%.4f × hours %@ %.4f)",
                        step.weight, step.intercept < 0 ? "−" : "+",
                        abs(step.intercept)))
                .font(.callout.monospaced()).foregroundStyle(Color.accentColor)
            Text("crosses 0.5 at \(boundaryText)")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - Controls

    private var learningRatePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Learning rate").font(.caption)
                Spacer()
                Text(String(format: "η = %g", export.runs[runIndex].learningRate))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Picker("Learning rate", selection: $runIndex) {
                ForEach(export.runs.indices, id: \.self) { i in
                    Text(String(format: "%g", export.runs[i].learningRate)).tag(i)
                }
            }
            .pickerStyle(.segmented)
            Text("Cross-entropy cannot explode the way squared error does — the "
                 + "error term is never bigger than 1 — so a large rate spoils the "
                 + "fit rather than producing infinities.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func playback(run: LogisticRun, step: LogisticFrame, last: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 14) {
                Button {
                    if Int(frame) >= last { frame = 0 }
                    playing.toggle()
                } label: {
                    Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 32))
                }
                Button { playing = false; frame = 0 } label: {
                    Image(systemName: "backward.end.fill").font(.title3)
                }
                Spacer()
                Text("iteration \(step.iter) / \(run.iterations)")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $frame, in: 0...Double(max(last, 1)), step: 1)
        }
    }

    // MARK: - Scores

    private func scores(_ step: LogisticFrame) -> some View {
        HStack(spacing: 10) {
            card("log-loss", String(format: "%.4f", step.logLoss))
            card("train acc", String(format: "%.1f%%", step.accuracy * 100))
            card("test acc", String(format: "%.1f%%", step.testAccuracy * 100))
        }
    }

    private func card(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func section<Content: View>(_ title: String,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
    }
}

// MARK: - The curve being fitted

/// The students, the curve as it stands, and the converged curve behind it.
struct FittingCurveView: View {
    let export: LogisticExport
    let current: LogisticFrame

    var body: some View {
        Canvas { context, size in
            let d = export.dataset
            let xLo = export.curve.first!.x, xHi = export.curve.last!.x
            let pad: CGFloat = 22

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - xLo) / (xHi - xLo)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat(v) * (size.height - 2 * pad)
            }

            for level in [0.0, 0.5, 1.0] {
                var grid = Path()
                grid.move(to: CGPoint(x: pad, y: sy(level)))
                grid.addLine(to: CGPoint(x: size.width - pad, y: sy(level)))
                context.stroke(grid, with: .color(.gray.opacity(level == 0.5 ? 0.45 : 0.22)),
                               style: StrokeStyle(lineWidth: 1, dash: level == 0.5 ? [4, 3] : []))
            }

            for (i, point) in d.points.enumerated() {
                let wobble = (Double((i * 37) % 11) / 11.0 - 0.5) * 0.10
                let c = CGPoint(x: sx(point.x[0]), y: sy(Double(point.y) + wobble))
                context.fill(Path(ellipseIn: CGRect(x: c.x - 2.5, y: c.y - 2.5,
                                                    width: 5, height: 5)),
                             with: .color((point.y == 1 ? Color.green : .red).opacity(0.4)))
            }

            // Where training will end up.
            var target = Path()
            for (i, sample) in export.curve.enumerated() {
                let point = CGPoint(x: sx(sample.x), y: sy(sample.p))
                if i == 0 { target.move(to: point) } else { target.addLine(to: point) }
            }
            context.stroke(target, with: .color(.gray.opacity(0.55)),
                           style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))

            // Where it is now.
            var live = Path()
            for (i, sample) in export.curve.enumerated() {
                let point = CGPoint(x: sx(sample.x), y: sy(current.probability(sample.x)))
                if i == 0 { live.move(to: point) } else { live.addLine(to: point) }
            }
            context.stroke(live, with: .color(.accentColor), lineWidth: 3)

            if let boundary = current.boundary, boundary > xLo, boundary < xHi {
                var cut = Path()
                cut.move(to: CGPoint(x: sx(boundary), y: pad))
                cut.addLine(to: CGPoint(x: sx(boundary), y: size.height - pad))
                context.stroke(cut, with: .color(.primary.opacity(0.35)),
                               style: StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
            }
        }
    }
}

// MARK: - Log-loss over iterations

/// The training loss against iteration, with the best achievable as a floor.
struct LogLossCurveView: View {
    let history: [LogisticFrame]
    let best: Double
    let currentIteration: Int

    var body: some View {
        Canvas { context, size in
            guard history.count > 1 else { return }
            let losses = history.map(\.logLoss)
            let lo = min(losses.min()!, best) * 0.95
            let hi = losses.max()! * 1.05
            let maxIter = Double(history.last?.iter ?? 1)

            func sx(_ it: Double) -> CGFloat { CGFloat(it / max(maxIter, 1)) * size.width }
            func sy(_ v: Double) -> CGFloat {
                size.height - CGFloat((v - lo) / max(hi - lo, 1e-9)) * size.height
            }

            var floorLine = Path()
            floorLine.move(to: CGPoint(x: 0, y: sy(best)))
            floorLine.addLine(to: CGPoint(x: size.width, y: sy(best)))
            context.stroke(floorLine, with: .color(.gray.opacity(0.6)),
                           style: StrokeStyle(lineWidth: 1, dash: [4, 3]))

            var curve = Path()
            for (i, frame) in history.enumerated() {
                let point = CGPoint(x: sx(Double(frame.iter)), y: sy(frame.logLoss))
                if i == 0 { curve.move(to: point) } else { curve.addLine(to: point) }
            }
            context.stroke(curve, with: .color(.accentColor), lineWidth: 2)

            var marker = Path()
            marker.move(to: CGPoint(x: sx(Double(currentIteration)), y: 0))
            marker.addLine(to: CGPoint(x: sx(Double(currentIteration)), y: size.height))
            context.stroke(marker, with: .color(.primary.opacity(0.35)), lineWidth: 1)
        }
    }
}

// MARK: - The loss landscape

/// The cross-entropy over every (w, b), with the path training has taken.
struct LossLandscapeView: View {
    let surface: LossSurface
    let history: [LogisticFrame]
    let upTo: Int
    let target: [Double]

    var body: some View {
        Canvas { context, size in
            let pad: CGFloat = 18
            // Most of the grid sits close to the minimum, so a linear shading
            // would compress the basin into a couple of tones. Ranking on a log
            // scale spreads the interesting region across the whole palette.
            let logLo = log(max(surface.lowest, 1e-6))
            let logHi = log(max(surface.highest, 1e-6))

            func sx(_ w: Double) -> CGFloat {
                pad + CGFloat((w - surface.wMin) / (surface.wMax - surface.wMin))
                    * (size.width - 2 * pad)
            }
            func sy(_ b: Double) -> CGFloat {
                size.height - pad
                    - CGFloat((b - surface.bMin) / (surface.bMax - surface.bMin))
                    * (size.height - 2 * pad)
            }

            // The landscape itself, one cell per sampled pair.
            let cellW = (size.width - 2 * pad) / CGFloat(surface.steps - 1)
            let cellH = (size.height - 2 * pad) / CGFloat(surface.steps - 1)
            for (bIndex, row) in surface.values.enumerated() {
                for (wIndex, value) in row.enumerated() {
                    let t = (log(max(value, 1e-6)) - logLo) / max(logHi - logLo, 1e-9)
                    let w = surface.wMin + (surface.wMax - surface.wMin)
                        * Double(wIndex) / Double(surface.steps - 1)
                    let b = surface.bMin + (surface.bMax - surface.bMin)
                        * Double(bIndex) / Double(surface.steps - 1)
                    let rect = CGRect(x: sx(w) - cellW / 2, y: sy(b) - cellH / 2,
                                      width: cellW + 1, height: cellH + 1)
                    // Dark where the loss is low, bright where it is high.
                    context.fill(Path(rect),
                                 with: .color(Color(hue: 0.62 - 0.45 * t,
                                                    saturation: 0.75,
                                                    brightness: 0.35 + 0.6 * t)))
                }
            }

            // The route taken so far.
            var path = Path()
            for i in 0...max(upTo, 0) where i < history.count {
                let theta = history[i].theta
                let point = CGPoint(x: sx(theta[0]), y: sy(theta[1]))
                if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
            }
            context.stroke(path, with: .color(.white.opacity(0.9)), lineWidth: 1.3)

            // The best fit, and where the run currently stands. Both get a dark
            // ring so they stay legible over the pale, high-loss areas.
            func marker(_ point: CGPoint, fill: Color, radius: CGFloat) {
                let rect = CGRect(x: point.x - radius, y: point.y - radius,
                                  width: 2 * radius, height: 2 * radius)
                context.fill(Path(ellipseIn: rect), with: .color(fill))
                context.stroke(Path(ellipseIn: rect),
                               with: .color(.black.opacity(0.75)), lineWidth: 1.5)
            }
            marker(CGPoint(x: sx(target[0]), y: sy(target[1])), fill: .yellow, radius: 5.5)
            if upTo < history.count {
                let theta = history[upTo].theta
                marker(CGPoint(x: sx(theta[0]), y: sy(theta[1])), fill: .white, radius: 4.5)
            }

            context.draw(Text("w →").font(.caption2).foregroundStyle(.white.opacity(0.85)),
                         at: CGPoint(x: size.width - pad, y: size.height - 4),
                         anchor: .bottomTrailing)
            context.draw(Text("b ↑").font(.caption2).foregroundStyle(.white.opacity(0.85)),
                         at: CGPoint(x: pad + 2, y: pad), anchor: .topLeading)
        }
    }
}
