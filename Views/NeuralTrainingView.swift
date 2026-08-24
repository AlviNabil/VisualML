//
//  NeuralTrainingView.swift
//  VisualML
//
//  Step 6 of the neural network walkthrough: forward and backward repeated
//  until the boundary fits. The map shows what the network predicts everywhere
//  on the plane, so the straight cuts of an untrained model can be watched
//  bending into a closed loop around the inner class.
//

import SwiftUI
import Combine

struct NeuralTrainingView: View {
    let export: NeuralExport

    @State private var frame: Double = 0
    @State private var playing = false
    @State private var showNeurons = false

    private let timer = Timer.publish(every: 0.35, on: .main, in: .common).autoconnect()

    var body: some View {
        let frames = export.training.frames
        let last = max(frames.count - 1, 0)
        let current = frames[min(Int(frame), last)]

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                NeuralScatterView(export: export, grid: current.grid)
                    .frame(height: 300)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                mapLegend
                playback(current: current, last: last)
                scores(current)

                section("The loss falling") {
                    NeuralLossCurveView(curve: export.training.curve,
                                        currentEpoch: current.epoch)
                        .frame(height: 130)
                }

                Toggle("Show what each hidden unit learned", isOn: $showNeurons.animation())
                    .font(.subheadline)
                if showNeurons { neuronGrid }

                closing
            }
            .padding()
        }
        .navigationTitle("6 · Training")
        .navigationBarTitleDisplayMode(.inline)
        .neuralInfo(export, topic: .training)
        .onReceive(timer) { _ in
            guard playing else { return }
            if Int(frame) >= last { playing = false }
            else { frame = min(frame + 1, Double(last)) }
        }
    }

    // MARK: - Copy

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Repeat until it fits").font(.headline)
            Text("One epoch is a forward pass over every point, then one "
                 + "backward pass, then a small step for every weight. The "
                 + "shading is what the network predicts across the whole plane; "
                 + "the dark line is where it changes its mind.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var mapLegend: some View {
        HStack(spacing: 14) {
            HStack(spacing: 5) {
                Rectangle().fill(.blue.opacity(0.5)).frame(width: 12, height: 10)
                Text(export.dataset.classNames[0]).font(.caption2)
            }
            HStack(spacing: 5) {
                Rectangle().fill(.orange.opacity(0.5)).frame(width: 12, height: 10)
                Text(export.dataset.classNames[1]).font(.caption2)
            }
            HStack(spacing: 5) {
                Circle().fill(.primary.opacity(0.55)).frame(width: 8, height: 8)
                Text("boundary").font(.caption2)
            }
            Spacer()
        }
        .foregroundStyle(.secondary)
    }

    private func playback(current: TrainingFrame, last: Int) -> some View {
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
                Text("epoch \(current.epoch) / \(export.architecture.epochs)")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $frame, in: 0...Double(max(last, 1)), step: 1)
        }
    }

    private func scores(_ current: TrainingFrame) -> some View {
        HStack(spacing: 10) {
            stat("loss", String(format: "%.4f", current.loss))
            stat("accuracy", String(format: "%.1f%%", current.accuracy * 100))
            stat("a line manages",
                 String(format: "%.1f%%", export.dataset.linearBaselineAccuracy * 100))
        }
    }

    // MARK: - Hidden units

    private var neuronGrid: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Each unit ends up watching a different direction. The first "
                 + "layer draws straight cuts; the second combines them, and the "
                 + "combination is what closes the loop.")
                .font(.caption).foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 78), spacing: 8)], spacing: 8) {
                ForEach(export.neurons) { neuron in
                    VStack(spacing: 3) {
                        NeuronResponseView(neuron: neuron)
                            .frame(height: 78)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        Text("L\(neuron.layer)·u\(neuron.unit + 1)")
                            .font(.caption2.monospaced()).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var closing: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("What just happened").font(.headline)
            Text(String(format: "Starting from random weights the network went "
                        + "from roughly chance to %.1f%%, on a problem where the "
                        + "best possible straight boundary reaches %.1f%%. "
                        + "Nothing in the algorithm knows anything about circles "
                        + "— the shape came out of repeatedly nudging %d numbers "
                        + "downhill.",
                        export.training.finalAccuracy * 100,
                        export.dataset.linearBaselineAccuracy * 100,
                        export.architecture.parameterCount))
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.green.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Pieces

    private func section<Content: View>(_ title: String,
                                        @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
                .padding(8)
                .background(Color.gray.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(title).font(.caption2).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Loss curve

/// Training loss against epoch, on a log-log scale with the current position
/// marked.
struct NeuralLossCurveView: View {
    let curve: [LossPoint]
    let currentEpoch: Int

    var body: some View {
        Canvas { context, size in
            guard curve.count > 1 else { return }
            let losses = curve.map(\.loss)
            guard let lo = losses.min(), let hi = losses.max(), hi > lo else { return }
            let maxEpoch = Double(curve.last?.epoch ?? 1)

            func sx(_ epoch: Double) -> CGFloat {
                CGFloat(log10(epoch + 1) / log10(maxEpoch + 1)) * size.width
            }
            func sy(_ loss: Double) -> CGFloat {
                let t = (log10(Swift.max(loss, 1e-6)) - log10(Swift.max(lo, 1e-6)))
                    / (log10(hi) - log10(Swift.max(lo, 1e-6)))
                return size.height - CGFloat(t) * size.height
            }

            var path = Path()
            for (index, point) in curve.enumerated() {
                let screen = CGPoint(x: sx(Double(point.epoch)), y: sy(point.loss))
                if index == 0 { path.move(to: screen) } else { path.addLine(to: screen) }
            }
            context.stroke(path, with: .color(.accentColor), lineWidth: 2)

            var marker = Path()
            marker.move(to: CGPoint(x: sx(Double(currentEpoch)), y: 0))
            marker.addLine(to: CGPoint(x: sx(Double(currentEpoch)), y: size.height))
            context.stroke(marker, with: .color(.primary.opacity(0.35)), lineWidth: 1)
        }
    }
}

// MARK: - One hidden unit's response

/// How strongly one hidden unit fires across the plane.
struct NeuronResponseView: View {
    let neuron: NeuronResponse

    var body: some View {
        Canvas { context, size in
            let grid = neuron.grid
            let steps = grid.count
            guard steps > 1 else { return }
            let cell = size.width / CGFloat(steps - 1)
            let cellY = size.height / CGFloat(steps - 1)

            for (row, values) in grid.enumerated() {
                for (column, value) in values.enumerated() {
                    let rect = CGRect(x: CGFloat(column) * cell - cell / 2,
                                      y: size.height - CGFloat(row) * cellY - cellY / 2,
                                      width: cell + 1, height: cellY + 1)
                    // Purple where the unit is negative, orange where positive.
                    let magnitude = Swift.min(abs(value), 1.0)
                    let colour = (value >= 0 ? Color.orange : Color.purple)
                        .opacity(0.12 + 0.75 * magnitude)
                    context.fill(Path(rect), with: .color(colour))
                }
            }
        }
    }
}
