//
//  NeuralBackpropView.swift
//  VisualML
//
//  Step 5 of the neural network walkthrough: the error travelling backward.
//  Each layer receives a signal from the one above it, multiplies it by its own
//  activation slope, works out how its weights should change, and passes what
//  is left further back. Every number here is the gradient the network actually
//  computed, on a partly-trained model where the error is still large.
//

import SwiftUI

struct NeuralBackpropView: View {
    let export: NeuralExport

    /// How many layers back the signal has travelled. 0 shows the output error.
    @State private var stage = 0

    var body: some View {
        let trace = export.backwardTrace
        let total = trace.steps.count

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro(trace)

                NetworkDiagramView(layerSizes: export.architecture.layerSizes,
                                   nodeValues: gradientMagnitudes(upTo: stage),
                                   nodeScale: .magnitude(.red),
                                   reversed: true,
                                   focusLayer: focusLayer(for: stage),
                                   showValues: false)
                    .frame(height: 240)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                gradientLegend
                stageControl(total: total)

                if stage == 0 {
                    startCard(trace)
                } else {
                    layerCard(trace.steps[stage - 1], index: stage - 1)
                }

                if stage == total { shrinkSummary(trace) }
            }
            .padding()
        }
        .navigationTitle("5 · Backpropagation")
        .navigationBarTitleDisplayMode(.inline)
        .neuralInfo(export, topic: .backprop)
    }

    // MARK: - Copy

    private func intro(_ trace: BackwardTrace) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("The error, travelling backward").font(.headline)
            Text(String(format: "Taken at epoch %d, while the network is still "
                        + "wrong: it says %.4f for a point whose true class is "
                        + "%d. That gap is what gets sent back.",
                        trace.epoch, trace.prediction, trace.label))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var gradientLegend: some View {
        HStack(spacing: 14) {
            HStack(spacing: 5) {
                Circle().fill(.red.opacity(0.7)).frame(width: 9, height: 9)
                Text("stronger gradient").font(.caption2)
            }
            HStack(spacing: 5) {
                Circle().fill(.red.opacity(0.15)).frame(width: 9, height: 9)
                Text("weaker").font(.caption2)
            }
            Spacer()
            Text("← direction of travel").font(.caption2)
        }
        .foregroundStyle(.secondary)
    }

    private func stageControl(total: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Travelling back").font(.caption)
                Spacer()
                Text(stage == 0 ? "at the output" : "through layer \(total - stage + 1)")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Picker("Stage", selection: $stage) {
                Text("err").tag(0)
                ForEach(1...total, id: \.self) { index in
                    Text("L\(total - index + 1)").tag(index)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    // MARK: - Cards

    private func startCard(_ trace: BackwardTrace) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Where it starts").font(.headline)
            Text("Pairing a sigmoid output with cross-entropy loss makes the "
                 + "first gradient remarkably simple — the two derivatives cancel "
                 + "and what is left is just the miss:")
                .font(.caption).foregroundStyle(.secondary)
            formula("dL/dz = prediction − label")
            formula(String(format: "      = %.4f − %d = %+.4f",
                           trace.prediction, trace.label, trace.outputError))
            Text("A positive value means the network guessed too high, a "
                 + "negative one too low. Everything that follows is this one "
                 + "number being shared out among the weights that caused it.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func layerCard(_ step: BackwardStep, index: Int) -> some View {
        let forward = export.forwardTraceAtBackprop
        let incoming = forward.steps.first { $0.layer == step.layer }?.input ?? []
        return VStack(alignment: .leading, spacing: 12) {
            Text("Layer \(step.layer)").font(.headline)

            substep("1", "Error at this layer's sum", "dL/dz")
            valueRow(label: "dz", values: step.dz, tint: .red)

            substep("2", "How its weights should change", "dL/dW = aᵀ · dz")
            Text("Each weight's share is the value that came in through it, "
                 + "times the error it fed into.")
                .font(.caption2).foregroundStyle(.secondary)
            matrixBlock("dW  (\(step.dW.count) × \(step.dW.first?.count ?? 0))", step.dW)
            valueRow(label: "db", values: step.db, tint: .orange)

            if let daPrev = step.daPrev, let primeZ = step.primeZ {
                Divider()
                substep("3", "Hand it further back", "dL/da = dz · Wᵀ")
                valueRow(label: "da", values: daPrev, tint: .purple)

                substep("4", "Scale by the layer's own slope", "dz = da ⊙ f′(z)")
                valueRow(label: "f′(z)", values: primeZ, tint: .teal)
                chainExample(daPrev: daPrev, primeZ: primeZ)
            } else {
                Divider()
                Text("This is the first layer — its inputs are the raw features, "
                     + "so there is nothing further back to send.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if !incoming.isEmpty {
                Text("(the values that came in here on the way forward: "
                     + incoming.map { String(format: "%.3f", $0) }.joined(separator: ", ")
                     + ")")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    /// The chain rule made concrete on the first unit: the two numbers that get
    /// multiplied, and what comes out.
    private func chainExample(daPrev: [Double], primeZ: [Double]) -> some View {
        guard let da = daPrev.first, let slope = primeZ.first else {
            return AnyView(EmptyView())
        }
        let text = String(format: "unit 1:  %.4f  ×  %.4f  =  %.4f\n"
                          + "         arriving   slope      passed on",
                          da, slope, da * slope)
        return AnyView(
            Text(text)
                .font(.caption2.monospaced())
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        )
    }

    private func shrinkSummary(_ trace: BackwardTrace) -> some View {
        let strengths = trace.steps.map { step in
            step.dz.map(abs).max() ?? 0
        }
        return VStack(alignment: .leading, spacing: 10) {
            Text("What survived the journey").font(.headline)
            VStack(spacing: 5) {
                ForEach(trace.steps.indices, id: \.self) { index in
                    strengthRow(layer: trace.steps[index].layer,
                                strength: strengths[index],
                                peak: strengths.max() ?? 1)
                }
            }
            Text("The signal fades on the way back — every layer multiplies it "
                 + "by weights and by an activation slope, both usually smaller "
                 + "than one. With a squash whose slope never exceeds 0.25, this "
                 + "fading is what stops deep networks learning at all.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func strengthRow(layer: Int, strength: Double, peak: Double) -> some View {
        HStack(spacing: 8) {
            Text("layer \(layer)").font(.caption2.monospaced())
                .frame(width: 54, alignment: .leading).foregroundStyle(.secondary)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.gray.opacity(0.15))
                    Capsule().fill(Color.red.opacity(0.6))
                        .frame(width: geo.size.width * CGFloat(strength / max(peak, 1e-9)))
                }
            }
            .frame(height: 8)
            Text(String(format: "%.4f", strength))
                .font(.caption2.monospacedDigit())
                .frame(width: 56, alignment: .trailing)
        }
    }

    // MARK: - Diagram feeding

    /// Gradient strength per layer, revealed as the signal travels back.
    private func gradientMagnitudes(upTo stage: Int) -> [[Double]] {
        let trace = export.backwardTrace
        let layerCount = export.architecture.layerSizes.count
        var values = [[Double]](repeating: [], count: layerCount)

        // steps are ordered output-first; step i covers layer (total - i).
        for (index, step) in trace.steps.enumerated() where index < stage {
            values[step.layer] = step.dz.map(abs)
        }
        if stage == 0, let first = trace.steps.first {
            values[first.layer] = first.dz.map(abs)
        }
        return values
    }

    private func focusLayer(for stage: Int) -> Int? {
        let total = export.backwardTrace.steps.count
        guard stage > 0 else { return export.architecture.layerSizes.count - 1 }
        return export.backwardTrace.steps[stage - 1].layer
    }

    // MARK: - Pieces

    private func substep(_ number: String, _ title: String, _ formula: String) -> some View {
        HStack(spacing: 8) {
            Text(number)
                .font(.caption2.bold())
                .frame(width: 18, height: 18)
                .background(Color.secondary.opacity(0.2), in: Circle())
            Text(title).font(.subheadline.bold())
            Spacer()
            Text(formula).font(.caption2.monospaced()).foregroundStyle(.secondary)
        }
    }

    private func valueRow(label: String, values: [Double], tint: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.caption2.monospaced()).foregroundStyle(.secondary)
                .frame(width: 38, alignment: .leading)
            FlowLayout(spacing: 5) {
                ForEach(values.indices, id: \.self) { index in
                    Text(String(format: "%+.4f", values[index]))
                        .font(.caption2.monospacedDigit())
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(tint)
                }
            }
        }
    }

    private func matrixBlock(_ title: String, _ rows: [[Double]]) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                ForEach(rows.indices, id: \.self) { row in
                    HStack(spacing: 6) {
                        ForEach(rows[row].indices, id: \.self) { column in
                            Text(String(format: "%+.3f", rows[row][column]))
                                .font(.caption2.monospacedDigit())
                                .frame(minWidth: 50, alignment: .trailing)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(8)
            .background(Color.gray.opacity(0.10))
            .clipShape(RoundedRectangle(cornerRadius: 5))
        }
    }

    private func formula(_ text: String) -> some View {
        Text(text)
            .font(.caption.monospaced())
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
