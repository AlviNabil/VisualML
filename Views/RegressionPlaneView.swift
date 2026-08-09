//
//  RegressionPlaneView.swift
//  VisualML
//
//  Two-feature linear regression. With two inputs the fit is no longer a line
//  but a plane through a 3-D cloud, so the scene is projected by hand and can be
//  rotated by dragging. All parameters come from the precomputed JSON produced
//  by ml/make_linear.py.
//

import SwiftUI
import Combine

// MARK: - 3-D projection

/// Orthographic projection of a point in the unit cube onto the screen.
///
/// The scene is rotated by `yaw` about the vertical axis and then by `pitch`
/// about the horizontal one. `depth` is returned so callers can sort what they
/// draw from back to front.
struct Projector3D {
    var yaw: Double
    var pitch: Double
    var scale: CGFloat
    var center: CGPoint

    func project(x: Double, y: Double, z: Double) -> (point: CGPoint, depth: Double) {
        let cy = cos(yaw), sy = sin(yaw)
        let x1 = x * cy + z * sy
        let z1 = -x * sy + z * cy

        let cp = cos(pitch), sp = sin(pitch)
        let y1 = y * cp - z1 * sp
        let depth = y * sp + z1 * cp

        return (CGPoint(x: center.x + CGFloat(x1) * scale,
                        y: center.y - CGFloat(y1) * scale), depth)
    }
}

// MARK: - The screen

struct RegressionPlaneView: View {
    @State private var export: RegressionExport?
    @State private var loadError: String?
    @State private var runIndex = 3
    @State private var frame: Double = 0
    @State private var playing = false
    @State private var yaw: Double = 0.7
    @State private var pitch: Double = 0.35
    @State private var dragStart: CGSize = .zero
    @State private var showPlane = true
    @State private var showInfo = false

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
        .navigationTitle("Multiple Regression")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button { showInfo = true } label: { Image(systemName: "info.circle") }
                    .accessibilityLabel("The maths behind this fit")
                    .disabled(export == nil)
            }
        }
        .sheet(isPresented: $showInfo) {
            if let export {
                RegressionInfoSheet(export: export)
                    .presentationDetents([.large])
            }
        }
        .task {
            do { export = try RegressionExport.load("linear_regression_multi") }
            catch { loadError = error.localizedDescription }
        }
    }

    private func content(_ export: RegressionExport) -> some View {
        let run = export.runs[runIndex]
        let last = max(run.history.count - 1, 0)
        let step = run.history[min(Int(frame), last)]

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                header(export)
                scene(export, step)
                rotationControls
                equations(export, step)
                learningRatePicker(export)
                playback(run: run, step: step, last: last)
                GradientStepExplainer(export: export, frame: step,
                                      learningRate: run.learningRate)
                comparison(export, step)
                dataTableLink(export)
            }
            .padding()
        }
        .onReceive(timer) { _ in
            guard playing else { return }
            if Int(frame) >= last { playing = false }
            else { frame = min(frame + 1, Double(last)) }
        }
        .onChange(of: runIndex) { _, _ in frame = 0; playing = false }
    }

    private func header(_ export: RegressionExport) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(export.subtitle).font(.headline)
            Text("With two inputs the fit becomes a plane: height is the predicted "
                 + "score, and the two floor axes are the features. Drag the scene "
                 + "to rotate it.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - The 3-D scene

    private func scene(_ export: RegressionExport, _ step: RegressionFrame) -> some View {
        PlaneSceneView(export: export, current: step,
                       yaw: yaw, pitch: pitch, showPlane: showPlane)
            .frame(height: 330)
            .background(Color.gray.opacity(0.10))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .gesture(
                DragGesture()
                    .onChanged { value in
                        let dx = value.translation.width - dragStart.width
                        let dy = value.translation.height - dragStart.height
                        dragStart = value.translation
                        yaw += Double(dx) * 0.012
                        pitch = max(-1.4, min(1.4, pitch + Double(dy) * 0.012))
                    }
                    .onEnded { _ in dragStart = .zero }
            )
    }

    private var rotationControls: some View {
        HStack(spacing: 12) {
            Image(systemName: "hand.draw").foregroundStyle(.secondary)
            Text("Drag to rotate").font(.caption).foregroundStyle(.secondary)
            Spacer()
            Toggle("Plane", isOn: $showPlane).labelsHidden()
            Text("Plane").font(.caption)
            Button("Reset") { yaw = 0.7; pitch = 0.35 }
                .font(.caption)
        }
    }

    // MARK: - Equations

    private func equations(_ export: RegressionExport, _ step: RegressionFrame) -> some View {
        let names = export.dataset.axisLabels.map { $0.lowercased() }
        let exact = String(format: "%.3f × %@ + %.3f × %@ + %.3f",
                           export.closedForm.weights[0], names[0],
                           export.closedForm.weights[1], names[1],
                           export.closedForm.intercept)
        let now = String(format: "%.3f × %@ + %.3f × %@ + %.3f",
                         step.weights[0], names[0],
                         step.weights[1], names[1], step.intercept)
        return VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Closed form").font(.caption2).foregroundStyle(.secondary)
                Text(exact).font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Gradient descent").font(.caption2).foregroundStyle(.secondary)
                Text(now).font(.caption.monospaced()).foregroundStyle(Color.accentColor)
            }
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
        }
    }

    private func playback(run: RegressionRun, step: RegressionFrame, last: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if run.diverged {
                Label("This rate diverges — the plane never settles.",
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

    // MARK: - Score comparison

    private func comparison(_ export: RegressionExport, _ step: RegressionFrame) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Scores").font(.headline)
            HStack(spacing: 10) {
                card("R² train", formatScore(step.r2))
                card("R² test", formatScore(step.testR2))
                card("RMSE", formatScore(step.mse.squareRoot()))
            }
            Text(String(format: "Adding a second feature lifts R² from about 0.50 "
                        + "(hours alone) to %.2f here — the same target, predicted "
                        + "from more information.", export.closedForm.r2))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func dataTableLink(_ export: RegressionExport) -> some View {
        NavigationLink {
            RegressionDataTableView(export: export, solution: export.closedForm)
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "tablecells")
                Text("Browse the data").fontWeight(.semibold)
                Spacer()
                Text("\(export.dataset.n) rows").font(.caption)
                Image(systemName: "chevron.right").font(.footnote.weight(.bold))
            }
            .padding(.vertical, 12).padding(.horizontal, 16)
            .frame(maxWidth: .infinity)
            .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
            .foregroundStyle(Color.accentColor)
        }
        .buttonStyle(.plain)
    }

    private func card(_ title: String, _ value: String) -> some View {
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

// MARK: - The rendered scene

/// Projects the 3-D point cloud and the fitted plane, drawing them back to front.
struct PlaneSceneView: View {
    let export: RegressionExport
    let current: RegressionFrame
    let yaw: Double
    let pitch: Double
    var showPlane: Bool = true

    /// Grid resolution of the plane mesh.
    private let steps = 10

    var body: some View {
        Canvas { context, size in
            let data = export.dataset
            guard data.mins.count >= 2 else { return }

            let projector = Projector3D(
                yaw: yaw, pitch: pitch,
                scale: min(size.width, size.height) * 0.34,
                center: CGPoint(x: size.width / 2, y: size.height / 2))

            // Map raw values onto [-1, 1] so the scene fits a fixed cube.
            func nx(_ v: Double) -> Double {
                2 * (v - data.mins[0]) / max(data.maxs[0] - data.mins[0], 1e-9) - 1
            }
            func nz(_ v: Double) -> Double {
                2 * (v - data.mins[1]) / max(data.maxs[1] - data.mins[1], 1e-9) - 1
            }
            func ny(_ v: Double) -> Double {
                2 * (v - data.targetMin) / max(data.targetMax - data.targetMin, 1e-9) - 1
            }

            var items: [(depth: Double, draw: (inout GraphicsContext) -> Void)] = []

            // Floor grid at the bottom of the cube.
            for i in 0...4 {
                let t = Double(i) / 4 * 2 - 1
                let a = projector.project(x: t, y: -1, z: -1)
                let b = projector.project(x: t, y: -1, z: 1)
                let c = projector.project(x: -1, y: -1, z: t)
                let d = projector.project(x: 1, y: -1, z: t)
                for (p, q) in [(a, b), (c, d)] {
                    var path = Path()
                    path.move(to: p.point); path.addLine(to: q.point)
                    items.append((min(p.depth, q.depth) - 10, { ctx in
                        ctx.stroke(path, with: .color(.gray.opacity(0.25)), lineWidth: 1)
                    }))
                }
            }

            // The fitted plane, as a mesh of quads.
            if showPlane {
                for i in 0..<steps {
                    for j in 0..<steps {
                        let u0 = Double(i) / Double(steps), u1 = Double(i + 1) / Double(steps)
                        let v0 = Double(j) / Double(steps), v1 = Double(j + 1) / Double(steps)

                        func corner(_ u: Double, _ v: Double) -> (CGPoint, Double) {
                            let f1 = data.mins[0] + u * (data.maxs[0] - data.mins[0])
                            let f2 = data.mins[1] + v * (data.maxs[1] - data.mins[1])
                            let yhat = current.predict([f1, f2])
                            let r = projector.project(x: nx(f1), y: ny(yhat), z: nz(f2))
                            return (r.point, r.depth)
                        }
                        let c00 = corner(u0, v0), c10 = corner(u1, v0)
                        let c11 = corner(u1, v1), c01 = corner(u0, v1)

                        var quad = Path()
                        quad.move(to: c00.0); quad.addLine(to: c10.0)
                        quad.addLine(to: c11.0); quad.addLine(to: c01.0)
                        quad.closeSubpath()

                        let depth = (c00.1 + c10.1 + c11.1 + c01.1) / 4
                        items.append((depth, { ctx in
                            ctx.fill(quad, with: .color(.accentColor.opacity(0.16)))
                            ctx.stroke(quad, with: .color(.accentColor.opacity(0.5)), lineWidth: 0.6)
                        }))
                    }
                }
            }

            // The students.
            for p in data.points {
                let r = projector.project(x: nx(p.x[0]), y: ny(p.y), z: nz(p.x[1]))
                let radius: CGFloat = p.test ? 4 : 3.4
                let rect = CGRect(x: r.point.x - radius, y: r.point.y - radius,
                                  width: 2 * radius, height: 2 * radius)
                let color: Color = p.test ? .green : .blue
                items.append((r.depth, { ctx in
                    ctx.fill(Path(ellipseIn: rect), with: .color(color.opacity(0.85)))
                }))
            }

            // Painter's algorithm: farthest first.
            for item in items.sorted(by: { $0.depth < $1.depth }) {
                var ctx = context
                item.draw(&ctx)
            }

            // Axis captions.
            context.draw(Text(data.axisLabels[0]).font(.caption2).foregroundStyle(.blue),
                         at: CGPoint(x: 10, y: size.height - 14), anchor: .leading)
            context.draw(Text(data.axisLabels[1]).font(.caption2).foregroundStyle(.orange),
                         at: CGPoint(x: size.width - 10, y: size.height - 14), anchor: .trailing)
            context.draw(Text(data.targetLabel + " ↑").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: 10, y: 12), anchor: .leading)
        }
    }
}
