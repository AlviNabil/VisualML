//
//  TrainingView.swift
//  VisualML
//
//  Replays gradient descent: the decision boundary starts as an arbitrary line
//  and you scrub / play through the recorded iterations to watch it converge
//  onto the data, with the training accuracy climbing alongside.
//

import SwiftUI
import Combine

struct TrainingView: View {
    @ObservedObject var viewModel: PipelineViewModel

    @State private var frame: Double = 0
    @State private var playing = false
    // ~14 fps; advances one recorded frame per tick while playing.
    private let timer = Timer.publish(every: 0.07, on: .main, in: .common).autoconnect()

    var body: some View {
        Group {
            if let lsa = viewModel.lsaResult, let model = viewModel.trainedModel,
               model.history.count > 1 {
                let last = model.history.count - 1
                let idx = min(Int(frame), last)
                let step = model.history[idx]

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        caption(model)
                        ScatterPlotView(coords: lsa.coords, labels: lsa.labels,
                                        boundary: (step.w0, step.w1, step.b),
                                        showMargins: model.drawMargins)
                            .frame(height: 320)
                            .padding(8)
                            .background(Color.gray.opacity(0.10))
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                        legend(lsa)
                        controls(idx: idx, last: last, step: step, model: model)
                    }
                    .padding()
                }
                .onReceive(timer) { _ in
                    guard playing else { return }
                    if Int(frame) >= last { playing = false }
                    else { frame = min(frame + 1, Double(last)) }
                }
            } else {
                ProgressView("Training…").frame(maxWidth: .infinity, minHeight: 220)
            }
        }
        .navigationTitle("Training, step by step")
        .navigationBarTitleDisplayMode(.inline)
        .task { await viewModel.computeLSA() }
    }

    // MARK: - Pieces

    private func caption(_ model: TrainedModel) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(model.kind.rawValue): fitting the boundary").font(.headline)
            Text("The line starts ARBITRARY (random weights) and each gradient-descent "
                 + "step nudges it to separate the classes. Press play, or drag the slider "
                 + "to step through it yourself.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func controls(idx: Int, last: Int, step: BoundaryStep, model: TrainedModel) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 14) {
                Button {
                    if Int(frame) >= last { frame = 0 }   // replay from the start
                    playing.toggle()
                } label: {
                    Image(systemName: playing ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 34))
                }
                Button { playing = false; frame = 0 } label: {
                    Image(systemName: "backward.end.fill").font(.title3)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("iteration \(step.iteration) / \(model.history.last?.iteration ?? 0)")
                        .font(.caption.monospacedDigit())
                    Text(String(format: "train accuracy %.0f%%", step.trainAccuracy * 100))
                        .font(.caption.bold().monospacedDigit())
                        .foregroundStyle(step.trainAccuracy > 0.85 ? .green : .primary)
                }
            }
            Slider(value: $frame, in: 0...Double(last), step: 1)
            Text(idx == 0 ? "Start: the arbitrary line."
                          : (idx == last ? "Converged: the final fitted boundary."
                                         : "Mid-training…"))
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

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
