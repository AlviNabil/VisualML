//
//  KMeansDataView.swift
//  VisualML
//
//  Step 1 of the k-means walkthrough: the raw, unlabeled dataset. There is no
//  target column here at all -- clustering has to find structure in the two
//  feature values alone.
//

import SwiftUI

struct KMeansDataView: View {
    let export: ClusterExport

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro
                ClusterMapView(dataset: export.dataset)
                    .frame(height: 300)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                stats
                whatToNotice
            }
            .padding()
        }
        .navigationTitle("1 · The data")
        .navigationBarTitleDisplayMode(.inline)
        .kmeansInfo(export, topic: .setup)
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("No labels this time").font(.headline)
            Text("Every earlier stage had a target: a score, a pass/fail. Here "
                 + "there is only \(export.dataset.n) customers' spending and how "
                 + "often they visit — nothing to predict, only structure to find.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var stats: some View {
        let d = export.dataset
        return HStack(spacing: 10) {
            stat("customers", "\(d.n)")
            stat(d.axisLabels[0], String(format: "%.0f–%.0f", d.mins[0], d.maxs[0]))
            stat(d.axisLabels[1], String(format: "%.1f–%.1f", d.mins[1], d.maxs[1]))
        }
    }

    private var whatToNotice: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("What to notice").font(.headline)
            Text("Even without colour, the cloud is not uniform — there are "
                 + "denser patches with gaps between them. That visual structure "
                 + "is exactly what k-means is about to go looking for.")
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
