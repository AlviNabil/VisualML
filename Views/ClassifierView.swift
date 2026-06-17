//
//  ClassifierView.swift
//  VisualML
//
//  Trains a linear classifier on the 2-D LSA points and draws its decision
//  boundary right on the scatter. Knobs (model, η, iterations, λ) retrain live.
//

import SwiftUI

struct ClassifierView: View {
    @ObservedObject var viewModel: PipelineViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let lsa = viewModel.lsaResult, let model = viewModel.trainedModel {
                    caption(model)
                    knobs
                    ScatterPlotView(coords: lsa.coords, labels: lsa.labels,
                                    boundary: (model.w0, model.w1, model.b),
                                    showMargins: model.drawMargins)
                        .frame(height: 320)
                        .padding(8)
                        .background(Color.gray.opacity(0.10))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    legend(lsa)
                    metrics(model)
                    confusionView(model)
                } else {
                    ProgressView("Training…")
                        .frame(maxWidth: .infinity, minHeight: 220)
                }
            }
            .padding()
        }
        .navigationTitle("Classifier")
        // LSA must be ready before we can train; cached so this is instant.
        .task { await viewModel.computeLSA() }
    }

    // MARK: - Caption

    private func caption(_ model: TrainedModel) -> some View {
        // Build the string outside the view builder — keeps the type-checker fast.
        let weighting = viewModel.config.weighting.rawValue
        let margin = model.drawMargins ? " Dashed lines are the SVM margins (z = ±1)." : ""
        let text = "A \(model.kind.rawValue) model fit on the 2-D LSA points (input: "
            + "\(weighting)). The solid line is where the score wᵀx + b = 0 — "
            + "\(model.class1.capitalized) on one side, \(model.class0.capitalized) "
            + "on the other." + margin
        return VStack(alignment: .leading, spacing: 4) {
            Text("Decision boundary").font(.headline)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - Knobs

    private var knobs: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Model", selection: $viewModel.classifierParams.kind) {
                ForEach(ClassifierKind.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            slider("Learning rate  η", $viewModel.classifierParams.learningRate,
                   1e-2...1.0, fmt: "%.2f")
            slider("Iterations", $viewModel.classifierParams.iterations,
                   50...1000, step: 50, fmt: "%.0f")
            slider("Regularization  λ", $viewModel.classifierParams.regularization,
                   0...0.1, fmt: "%.3f")
        }
    }

    private func slider(_ label: String, _ value: Binding<Double>,
                        _ range: ClosedRange<Double>, step: Double? = nil,
                        fmt: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: fmt, value.wrappedValue))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            if let step {
                Slider(value: value, in: range, step: step)
            } else {
                Slider(value: value, in: range)
            }
        }
    }

    // MARK: - Metrics

    private func metrics(_ model: TrainedModel) -> some View {
        HStack(spacing: 12) {
            metricCard("Train accuracy", model.trainAccuracy)
            metricCard("Test accuracy", model.testAccuracy)
        }
    }

    private func metricCard(_ title: String, _ value: Double) -> some View {
        VStack(spacing: 2) {
            Text(String(format: "%.0f%%", value * 100)).font(.title2).bold()
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Confusion matrix

    private func confusionView(_ model: TrainedModel) -> some View {
        let c = model.confusion
        return VStack(alignment: .leading, spacing: 8) {
            Text("Confusion matrix — test set (\(model.testCount) docs)").font(.headline)
            Grid(horizontalSpacing: 10, verticalSpacing: 6) {
                GridRow {
                    Text("").gridColumnAlignment(.leading)
                    Text("pred \(model.class0.capitalized)").font(.caption2).bold()
                    Text("pred \(model.class1.capitalized)").font(.caption2).bold()
                }
                GridRow {
                    Text("actual \(model.class0.capitalized)").font(.caption2).bold()
                    cell(c[0][0], correct: true); cell(c[0][1], correct: false)
                }
                GridRow {
                    Text("actual \(model.class1.capitalized)").font(.caption2).bold()
                    cell(c[1][0], correct: false); cell(c[1][1], correct: true)
                }
            }
            Text("Diagonal (green) = correct predictions; off-diagonal = mistakes.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func cell(_ n: Int, correct: Bool) -> some View {
        Text("\(n)")
            .font(.callout.monospacedDigit()).bold()
            .frame(width: 70, height: 40)
            .background((correct ? Color.green : Color.red).opacity(n > 0 ? 0.18 : 0.06))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Legend

    private func legend(_ lsa: SVDResult) -> some View {
        HStack(spacing: 16) {
            ForEach(classLegend(lsa), id: \.label) { item in
                HStack(spacing: 6) {
                    Circle().fill(color(forLabel: item.label)).frame(width: 10, height: 10)
                    Text(item.name.capitalized).font(.caption)
                }
            }
        }
    }

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    private func classLegend(_ lsa: SVDResult) -> [(name: String, label: Int)] {
        var seen: [Int: String] = [:]
        for (label, name) in zip(lsa.labels, lsa.categories) where seen[label] == nil {
            seen[label] = name
        }
        return seen.map { (name: $0.value, label: $0.key) }.sorted { $0.label < $1.label }
    }
}
