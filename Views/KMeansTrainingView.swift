//
//  KMeansTrainingView.swift
//  VisualML
//
//  Step 3 of the k-means walkthrough: watching the assign-then-average loop
//  run, from several different random starts. Most land in the same place;
//  one does not, which is the point -- k-means finds *a* good answer, not
//  provably *the* best one, and where it starts can decide where it ends up.
//

import SwiftUI
import Combine

struct KMeansTrainingView: View {
    let export: ClusterExport

    @State private var restartIndex = 0
    @State private var frame: Double = 0
    @State private var playing = false

    private let timer = Timer.publish(every: 0.35, on: .main, in: .common).autoconnect()

    var body: some View {
        let restart = export.restarts[restartIndex]
        let last = max(restart.history.count - 1, 0)
        let step = restart.history[min(Int(frame), last)]
        let isBest = restart.finalInertia <= export.best.inertia + 1e-6

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                ClusterMapView(dataset: export.dataset,
                               assignments: step.assignments,
                               centroids: step.centroids)
                    .frame(height: 290)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                restartPicker
                playback(restart: restart, step: step, last: last)
                outcome(restart: restart, isBest: isBest)
            }
            .padding()
        }
        .navigationTitle("3 · Finding the clusters")
        .navigationBarTitleDisplayMode(.inline)
        .kmeansInfo(export, topic: .algorithm)
        .onReceive(timer) { _ in
            guard playing else { return }
            if Int(frame) >= last { playing = false }
            else { frame = min(frame + 1, Double(last)) }
        }
        .onChange(of: restartIndex) { _, _ in frame = 0; playing = false }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Assign, then average").font(.headline)
            Text("Each round: colour every point by its nearest centroid (the X "
                 + "markers), then move each centroid to the average of the "
                 + "points now coloured that way. Repeat until nothing moves.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var restartPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Starting point").font(.caption)
                Spacer()
                Text("seed \(export.restarts[restartIndex].seed)")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Picker("Starting point", selection: $restartIndex) {
                ForEach(export.restarts.indices, id: \.self) { i in
                    Text("\(i + 1)").tag(i)
                }
            }
            .pickerStyle(.segmented)
            Text("Five different k-means++ starting points, same data, same k.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func playback(restart: ClusterRestart, step: ClusterFrame, last: Int) -> some View {
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
                Text("round \(step.iter) / \(restart.iterations)")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $frame, in: 0...Double(max(last, 1)), step: 1)
            Text(String(format: "inertia %.2f", step.inertia))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
    }

    private func outcome(restart: ClusterRestart, isBest: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Where it landed").font(.headline)
            HStack(spacing: 10) {
                stat("final inertia", String(format: "%.2f", restart.finalInertia))
                stat("rounds", "\(restart.iterations)")
                stat("matches truth", String(format: "%.0f%%", restart.purity * 100))
            }
            if isBest {
                Label("This is the best result across all five starts — the "
                      + "global optimum for this data.",
                      systemImage: "checkmark.circle.fill")
                    .font(.caption).foregroundStyle(.green)
            } else {
                Label("This start got stuck in a worse arrangement — notice the "
                      + "higher inertia and lower match to the true segments. "
                      + "Different starting centroids found a different, worse "
                      + "answer from the same data.",
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
            }
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
