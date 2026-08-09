//
//  LinearRegressionView.swift
//  VisualML
//
//  Single-feature linear regression: the scatter, the closed-form line, and a
//  replay of gradient descent walking onto it. Everything drawn here comes from
//  the precomputed JSON produced by ml/make_linear.py.
//

import SwiftUI
import Combine

struct LinearRegressionView: View {
    @State private var export: RegressionExport?
    @State private var loadError: String?
    @State private var runIndex = 3
    @State private var frame: Double = 0
    @State private var playing = false
    @State private var showResiduals = false

    private let timer = Timer.publish(every: 0.06, on: .main, in: .common).autoconnect()

    var body: some View {
        Group {
            if let export {
                content(export)
            } else if let loadError {
                ContentUnavailableView("Could not load the model", systemImage: "exclamationmark.triangle",
                                       description: Text(loadError))
            } else {
                ProgressView("Loading…").frame(maxWidth: .infinity, minHeight: 240)
            }
        }
        .navigationTitle("Linear Regression")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            do { export = try RegressionExport.load("linear_regression") }
            catch { loadError = error.localizedDescription }
        }
    }

    // MARK: - Layout

    private func content(_ export: RegressionExport) -> some View {
        let run = export.runs[runIndex]
        let lastFrame = max(run.history.count - 1, 0)
        let step = run.history[min(Int(frame), lastFrame)]

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header(export)

                ScatterFitView(export: export, current: step,
                               showResiduals: showResiduals)
                    .frame(height: 300)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                legend
                Toggle("Show residuals", isOn: $showResiduals)
                    .font(.subheadline)

                equations(export, step)
                learningRatePicker(export)
                playback(run: run, step: step, last: lastFrame)
                lossCurve(run: run, export: export, current: step)
                metrics(export, step)
            }
            .padding()
        }
        .onReceive(timer) { _ in
            guard playing else { return }
            if Int(frame) >= lastFrame { playing = false }
            else { frame = min(frame + 1, Double(lastFrame)) }
        }
        .onChange(of: runIndex) { _, _ in frame = 0; playing = false }
    }

    private func header(_ export: RegressionExport) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(export.subtitle).font(.headline)
            Text("\(export.dataset.n) students · \(export.dataset.trainCount) train / "
                 + "\(export.dataset.testCount) test. The dashed line is the exact "
                 + "least-squares answer; the solid line is where gradient descent "
                 + "has reached.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var legend: some View {
        HStack(spacing: 16) {
            Label { Text("train").font(.caption) } icon: {
                Circle().fill(.blue).frame(width: 9, height: 9)
            }
            Label { Text("test").font(.caption) } icon: {
                Circle().fill(.green).frame(width: 9, height: 9)
            }
            Label { Text("closed form").font(.caption) } icon: {
                Rectangle().fill(.gray).frame(width: 14, height: 2)
            }
            Label { Text("gradient descent").font(.caption) } icon: {
                Rectangle().fill(Color.accentColor).frame(width: 14, height: 3)
            }
        }
    }

    // MARK: - Equations

    private func equations(_ export: RegressionExport, _ step: RegressionFrame) -> some View {
        let name = export.dataset.axisLabels[0].lowercased()
        let exact = String(format: "%@ = %.3f × %@ + %.3f",
                           export.dataset.targetLabel.lowercased(),
                           export.closedForm.weights[0], name, export.closedForm.intercept)
        let now = String(format: "%@ = %.3f × %@ + %.3f",
                         export.dataset.targetLabel.lowercased(),
                         step.weights[0], name, step.intercept)
        return VStack(alignment: .leading, spacing: 6) {
            labelledEquation("Closed form", exact, .secondary)
            labelledEquation("Gradient descent", now, Color.accentColor)
        }
    }

    private func labelledEquation(_ title: String, _ text: String, _ tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            Text(text).font(.callout.monospaced()).foregroundStyle(tint)
        }
    }

    // MARK: - Controls

    private func learningRatePicker(_ export: RegressionExport) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Learning rate").font(.caption)
                Spacer()
                Text(String(format: "η = %.6f", export.runs[runIndex].learningRate))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Picker("Learning rate", selection: $runIndex) {
                ForEach(export.runs.indices, id: \.self) { i in
                    Text(export.runs[i].label).tag(i)
                }
            }
            .pickerStyle(.segmented)
            Text("Shown as a multiple of the largest stable step size. "
                 + "Above 1.00× the loss blows up.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func playback(run: RegressionRun, step: RegressionFrame, last: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if run.diverged {
                Label("This rate diverges — the loss runs away and training never settles.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.red)
            }
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

    // MARK: - Loss curve

    private func lossCurve(run: RegressionRun, export: RegressionExport,
                           current: RegressionFrame) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Training loss (MSE)").font(.headline)
            LossCurveView(history: run.history,
                          optimum: export.closedForm.mse,
                          currentIteration: current.iter)
                .frame(height: 120)
                .padding(8)
                .background(Color.gray.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - Metrics

    private func metrics(_ export: RegressionExport, _ step: RegressionFrame) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Scores").font(.headline)
            HStack(spacing: 10) {
                metricCard("R² train", formatScore(step.r2))
                metricCard("R² test", formatScore(step.testR2))
                metricCard("RMSE", formatScore(step.mse.squareRoot()))
            }
            let beaten = export.baseline.mse
            Text(String(format: "Predicting the mean (%.1f) for everyone scores MSE %.1f; "
                        + "the exact fit reaches %.1f.",
                        export.baseline.prediction, beaten, export.closedForm.mse))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func metricCard(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.title3.bold().monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.5)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Scatter + fitted line

/// Draws the data, the closed-form line and the current gradient-descent line.
struct ScatterFitView: View {
    let export: RegressionExport
    let current: RegressionFrame
    var showResiduals: Bool = false

    var body: some View {
        Canvas { context, size in
            let data = export.dataset
            let pad: CGFloat = 26
            let minX = data.mins[0], maxX = data.maxs[0]
            let minY = data.targetMin, maxY = data.targetMax

            func sx(_ v: Double) -> CGFloat {
                let t = (v - minX) / max(maxX - minX, 1e-9)
                return pad + CGFloat(t) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                let t = (v - minY) / max(maxY - minY, 1e-9)
                return size.height - pad - CGFloat(t) * (size.height - 2 * pad)
            }

            // Frame
            var box = Path()
            box.addRect(CGRect(x: pad, y: pad,
                               width: size.width - 2 * pad, height: size.height - 2 * pad))
            context.stroke(box, with: .color(.gray.opacity(0.25)), lineWidth: 1)

            // Residual sticks from each point to the current line.
            if showResiduals {
                var sticks = Path()
                for p in data.points {
                    let fitted = current.predict(p.x)
                    sticks.move(to: CGPoint(x: sx(p.x[0]), y: sy(p.y)))
                    sticks.addLine(to: CGPoint(x: sx(p.x[0]), y: sy(fitted)))
                }
                context.stroke(sticks, with: .color(.red.opacity(0.35)), lineWidth: 1)
            }

            // Points
            for p in data.points {
                let c = CGPoint(x: sx(p.x[0]), y: sy(p.y))
                let r: CGFloat = p.test ? 3.5 : 3
                let rect = CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)
                context.fill(Path(ellipseIn: rect),
                             with: .color(p.test ? .green.opacity(0.9) : .blue.opacity(0.65)))
            }

            // Lines: closed form (dashed) then the current GD line (solid).
            func line(_ solve: (Double) -> Double, color: Color, dash: Bool, width: CGFloat) {
                var path = Path()
                path.move(to: CGPoint(x: sx(minX), y: sy(solve(minX))))
                path.addLine(to: CGPoint(x: sx(maxX), y: sy(solve(maxX))))
                context.stroke(path, with: .color(color),
                               style: StrokeStyle(lineWidth: width, dash: dash ? [6, 4] : []))
            }
            line({ export.closedForm.predict([$0]) }, color: .gray, dash: true, width: 2)
            line({ current.predict([$0]) }, color: .accentColor, dash: false, width: 2.5)

            // Axis labels
            context.draw(Text(export.dataset.axisLabels[0]).font(.caption2)
                            .foregroundStyle(.secondary),
                         at: CGPoint(x: size.width / 2, y: size.height - 8))
            context.draw(Text(export.dataset.targetLabel).font(.caption2)
                            .foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: 10), anchor: .topLeading)
        }
    }
}

// MARK: - Loss curve

/// Plots MSE against iteration on a log scale, with the optimum as a floor.
struct LossCurveView: View {
    let history: [RegressionFrame]
    let optimum: Double
    let currentIteration: Int

    var body: some View {
        Canvas { context, size in
            guard history.count > 1 else { return }
            let losses = history.map { max($0.mse, 1e-6) }
            let logs = losses.map { log10($0) }
            let lo = min(logs.min()!, log10(max(optimum, 1e-6))) - 0.05
            let hi = logs.max()! + 0.05
            let maxIter = Double(history.last?.iter ?? 1)

            func sx(_ it: Double) -> CGFloat {
                CGFloat(it / max(maxIter, 1)) * size.width
            }
            func sy(_ logV: Double) -> CGFloat {
                size.height - CGFloat((logV - lo) / max(hi - lo, 1e-9)) * size.height
            }

            // The closed-form optimum: the floor the curve should approach.
            var floorLine = Path()
            let fy = sy(log10(max(optimum, 1e-6)))
            floorLine.move(to: CGPoint(x: 0, y: fy))
            floorLine.addLine(to: CGPoint(x: size.width, y: fy))
            context.stroke(floorLine, with: .color(.gray.opacity(0.6)),
                           style: StrokeStyle(lineWidth: 1, dash: [4, 3]))

            // The loss curve.
            var curve = Path()
            for (i, frame) in history.enumerated() {
                let p = CGPoint(x: sx(Double(frame.iter)), y: sy(logs[i]))
                if i == 0 { curve.move(to: p) } else { curve.addLine(to: p) }
            }
            context.stroke(curve, with: .color(.accentColor), lineWidth: 2)

            // Playback marker.
            var marker = Path()
            let mx = sx(Double(currentIteration))
            marker.move(to: CGPoint(x: mx, y: 0))
            marker.addLine(to: CGPoint(x: mx, y: size.height))
            context.stroke(marker, with: .color(.primary.opacity(0.35)), lineWidth: 1)
        }
    }
}
