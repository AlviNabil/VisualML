//
//  KMeansResultView.swift
//  VisualML
//
//  Step 4 of the k-means walkthrough: the map. The centroids are the model's
//  learned parameters -- the k-means equivalent of a regression's weights --
//  so they get listed explicitly, in the customers' own units, alongside the
//  coloured map and a check against the segments the data was generated from.
//

import SwiftUI

struct KMeansResultView: View {
    let export: ClusterExport

    @State private var showTruth = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                Group {
                    if showTruth {
                        TrueSegmentMapView(dataset: export.dataset)
                    } else {
                        ClusterMapView(dataset: export.dataset,
                                       assignments: export.best.assignments,
                                       centroids: export.best.centroids)
                    }
                }
                .frame(height: 300)
                .padding(8)
                .background(Color.gray.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: 10))

                if showTruth { truthLegend } else { legend }

                Toggle("Compare with the true segments", isOn: $showTruth.animation())
                    .font(.subheadline)

                weightsTable
                scores
            }
            .padding()
        }
        .navigationTitle("4 · The result")
        .navigationBarTitleDisplayMode(.inline)
        .kmeansInfo(export, topic: .result)
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("The map").font(.headline)
            Text("Every customer coloured by its nearest centroid, and the "
                 + "centroids themselves marked with ✕. No labels went into "
                 + "this — the groups came entirely from the two numbers per "
                 + "customer.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var legend: some View {
        HStack(spacing: 12) {
            ForEach(export.best.clusterLabels) { label in
                HStack(spacing: 5) {
                    Circle().fill(clusterColor(label.cluster)).frame(width: 9, height: 9)
                    Text(label.majoritySegment ?? "—").font(.caption2)
                }
            }
        }
    }

    private var truthLegend: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 12) {
                ForEach(Array(trueSegmentNames.enumerated()), id: \.offset) { i, name in
                    HStack(spacing: 5) {
                        Circle().fill(clusterColor(i)).frame(width: 9, height: 9)
                        Text(name).font(.caption2)
                    }
                }
            }
            Text("Shown here in the segment the data was generated from — "
                 + "k-means never saw this column. Compare the shapes above to "
                 + "check how closely the two colourings agree.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var trueSegmentNames: [String] {
        var seen: [String] = []
        for p in export.dataset.points where !seen.contains(p.trueSegment) { seen.append(p.trueSegment) }
        return seen.sorted()
    }

    // MARK: - The weights

    private var weightsTable: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("The centroids — k-means' \"weights\"").font(.headline)
            Text("A regression learns a coefficient per feature. k-means learns "
                 + "a whole point per cluster: the coordinates a typical member "
                 + "sits closest to. These four points are everything the model "
                 + "keeps.")
                .font(.caption).foregroundStyle(.secondary)

            VStack(spacing: 6) {
                headerRow
                ForEach(export.best.clusterLabels) { label in
                    centroidRow(label)
                }
            }
        }
    }

    private var headerRow: some View {
        HStack {
            Text("cluster").font(.caption2.bold()).frame(width: 70, alignment: .leading)
            Text(export.dataset.axisLabels[0]).font(.caption2.bold()).frame(maxWidth: .infinity, alignment: .trailing)
            Text(export.dataset.axisLabels[1]).font(.caption2.bold()).frame(maxWidth: .infinity, alignment: .trailing)
            Text("size").font(.caption2.bold()).frame(width: 40, alignment: .trailing)
        }
        .foregroundStyle(.secondary)
    }

    private func centroidRow(_ label: ClusterLabel) -> some View {
        let centroid = export.best.centroids[label.cluster]
        return HStack {
            HStack(spacing: 6) {
                Circle().fill(clusterColor(label.cluster)).frame(width: 8, height: 8)
                Text(label.majoritySegment ?? "cluster \(label.cluster)")
                    .font(.caption.bold())
            }
            .frame(width: 70, alignment: .leading)
            Text(String(format: "%.1f", centroid[0]))
                .font(.caption.monospacedDigit()).frame(maxWidth: .infinity, alignment: .trailing)
            Text(String(format: "%.1f", centroid[1]))
                .font(.caption.monospacedDigit()).frame(maxWidth: .infinity, alignment: .trailing)
            Text("\(label.size)")
                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.vertical, 4)
    }

    // MARK: - Scores

    private var scores: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("How well it did").font(.headline)
            HStack(spacing: 10) {
                stat("inertia", String(format: "%.1f", export.best.inertia))
                stat("matches truth", String(format: "%.0f%%", export.best.purity * 100))
                stat("k", "\(export.k)")
            }
            Text("\"Matches truth\" checks the clusters against the segments the "
                 + "data was generated from — a comparison only possible because "
                 + "this is simulated data. A real dataset has no such answer key; "
                 + "inertia and the elbow curve are all you would have.")
                .font(.caption).foregroundStyle(.secondary)
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

/// The customers coloured by the segment the data was generated from, used
/// only for the "compare with truth" toggle.
struct TrueSegmentMapView: View {
    let dataset: ClusterDataset

    private var segmentNames: [String] {
        var seen: [String] = []
        for p in dataset.points where !seen.contains(p.trueSegment) { seen.append(p.trueSegment) }
        return seen.sorted()
    }

    var body: some View {
        Canvas { context, size in
            let pad: CGFloat = 24
            let minX = dataset.mins[0], maxX = dataset.maxs[0]
            let minY = dataset.mins[1], maxY = dataset.maxs[1]
            let names = segmentNames

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - minX) / max(maxX - minX, 1e-9)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat((v - minY) / max(maxY - minY, 1e-9)) * (size.height - 2 * pad)
            }

            for point in dataset.points {
                let idx = names.firstIndex(of: point.trueSegment) ?? 0
                let c = CGPoint(x: sx(point.x[0]), y: sy(point.x[1]))
                let r: CGFloat = 3.2
                context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(clusterColor(idx).opacity(0.7)))
            }
        }
    }
}
