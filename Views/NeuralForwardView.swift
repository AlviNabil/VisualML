//
//  NeuralForwardView.swift
//  VisualML
//
//  Step 3 of the neural network walkthrough: one point pushed through the
//  network, one layer at a time. Every number is the one the model actually
//  produced -- the weighted sum before the squash, and the activation after
//  it -- so the whole forward pass can be followed arithmetically.
//

import SwiftUI

struct NeuralForwardView: View {
    let export: NeuralExport

    /// How many layers have been computed so far. 0 shows just the input.
    @State private var stage = 0

    var body: some View {
        let trace = export.forwardTrace
        let total = trace.steps.count

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro(trace)

                NetworkDiagramView(layerSizes: export.architecture.layerSizes,
                                   nodeValues: nodeValues(upTo: stage),
                                   weights: revealedWeights(upTo: stage),
                                   focusLayer: stage == 0 ? 0 : stage)
                    .frame(height: 250)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                stageControl(total: total)

                if stage == 0 {
                    inputCard(trace)
                } else {
                    layerCard(trace.steps[stage - 1])
                }

                if stage == total { verdict(trace) }
            }
            .padding()
        }
        .navigationTitle("3 · The forward pass")
        .navigationBarTitleDisplayMode(.inline)
        .neuralInfo(export, topic: .forward)
    }

    // MARK: - Copy

    private func intro(_ trace: ForwardTrace) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("One point, all the way through").font(.headline)
            Text(String(format: "Following the point (%.3f, %.3f), which really "
                        + "belongs to class %d. Step through the layers and watch "
                        + "it turn into a probability.",
                        trace.input[0], trace.input[1], trace.label))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func stageControl(total: Int) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Layer").font(.caption)
                Spacer()
                Text(stage == 0 ? "input" : "after layer \(stage)")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Picker("Layer", selection: $stage) {
                Text("in").tag(0)
                ForEach(1...total, id: \.self) { Text("\($0)").tag($0) }
            }
            .pickerStyle(.segmented)
        }
    }

    // MARK: - The cards

    private func inputCard(_ trace: ForwardTrace) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("The input").font(.headline)
            Text("Two numbers — the point's coordinates. Nothing has happened "
                 + "to them yet.")
                .font(.caption).foregroundStyle(.secondary)
            valueRow(label: "x", values: trace.input, tint: .blue)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func layerCard(_ step: ForwardStep) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Text("Layer \(step.layer)").font(.headline)
                Text(step.activation)
                    .font(.caption.monospaced())
                    .padding(.horizontal, 7).padding(.vertical, 3)
                    .background(Color.accentColor.opacity(0.15), in: Capsule())
                    .foregroundStyle(Color.accentColor)
            }

            substep("1", "Weighted sum", "z = a·W + b")
            valueRow(label: "a in", values: step.input, tint: .secondary)
            matrixBlock("W  (\(step.weights.count) × \(step.weights.first?.count ?? 0))",
                        step.weights)
            valueRow(label: "b", values: step.biases, tint: .secondary)
            valueRow(label: "z", values: step.z, tint: .purple)

            Divider()

            substep("2", "Squash", "a = \(step.activation)(z)")
            valueRow(label: "a out", values: step.a, tint: .accentColor)

            workedExample(step)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    /// Spell out the arithmetic for the first unit, so the matrix maths is
    /// grounded in one concrete number the reader can check.
    private func workedExample(_ step: ForwardStep) -> some View {
        let terms = step.input.indices.map { index -> String in
            String(format: "%.3f×%.3f", step.input[index], step.weights[index][0])
        }.joined(separator: " + ")
        let text = "unit 1:  z = " + terms
            + String(format: " + %.3f = %.3f", step.biases[0], step.z[0])
            + String(format: "\n         a = %@(%.3f) = %.3f",
                     step.activation, step.z[0], step.a[0])
        return Text(text)
            .font(.caption2.monospaced())
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func verdict(_ trace: ForwardTrace) -> some View {
        let predicted = trace.prediction >= 0.5 ? 1 : 0
        let correct = predicted == trace.label
        let names = export.dataset.classNames
        return VStack(alignment: .leading, spacing: 8) {
            Text("The answer").font(.headline)
            HStack(spacing: 10) {
                stat("output", String(format: "%.4f", trace.prediction))
                stat("predicts", names[predicted])
                stat("truth", names[trace.label])
                stat("loss", String(format: "%.4f", trace.loss))
            }
            Text(correct
                 ? "The single output unit gives the probability of the second "
                   + "class. Above 0.5 means that class — and here it matches the "
                   + "truth, with very little loss left."
                 : "The network got this one wrong, which is what the loss "
                   + "measures and what backpropagation will act on.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background((correct ? Color.green : Color.red).opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Diagram feeding

    /// Node values up to the current stage; later layers stay blank.
    private func nodeValues(upTo stage: Int) -> [[Double]] {
        let trace = export.forwardTrace
        var values: [[Double]] = [trace.input]
        for (index, step) in trace.steps.enumerated() {
            values.append(index < stage ? step.a : [])
        }
        return values
    }

    /// Only draw the connections that have been used so far.
    private func revealedWeights(upTo stage: Int) -> [[[Double]]]? {
        guard stage > 0 else { return nil }
        return export.forwardTrace.steps.prefix(stage).map(\.weights)
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
            Text(formula).font(.caption.monospaced()).foregroundStyle(.secondary)
        }
    }

    private func valueRow(label: String, values: [Double], tint: Color) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label)
                .font(.caption2.monospaced()).foregroundStyle(.secondary)
                .frame(width: 34, alignment: .leading)
            FlowLayout(spacing: 5) {
                ForEach(values.indices, id: \.self) { index in
                    Text(String(format: "%.3f", values[index]))
                        .font(.caption2.monospacedDigit())
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(tint == .secondary ? Color.secondary : tint)
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
                            Text(String(format: "%+.2f", rows[row][column]))
                                .font(.caption2.monospacedDigit())
                                .frame(minWidth: 44, alignment: .trailing)
                                .foregroundStyle(rows[row][column] >= 0 ? Color.blue : Color.red)
                        }
                    }
                }
            }
            .padding(8)
            .background(Color.gray.opacity(0.10))
            .clipShape(RoundedRectangle(cornerRadius: 5))
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
