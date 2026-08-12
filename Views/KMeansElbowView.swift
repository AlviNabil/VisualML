//
//  KMeansElbowView.swift
//  VisualML
//
//  Step 2 of the k-means walkthrough: choosing k. Adding a cluster always
//  lowers inertia -- with 180 clusters every point could sit exactly on its
//  own centroid -- so the question is not "what's the lowest inertia" but
//  "where does adding another cluster stop paying for itself".
//

import SwiftUI

struct KMeansElbowView: View {
    let export: ClusterExport

    @State private var selectedK: Int = 4

    var body: some View {
        let point = export.elbow.first { $0.k == selectedK } ?? export.elbow[0]

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                ElbowCurveView(elbow: export.elbow, selectedK: selectedK)
                    .frame(height: 160)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                kPicker

                Text("k = \(selectedK) on the map").font(.headline)
                ClusterMapView(dataset: export.dataset,
                               assignments: point.assignments,
                               centroids: point.centroids)
                    .frame(height: 280)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                reading
            }
            .padding()
        }
        .navigationTitle("2 · Choosing k")
        .navigationBarTitleDisplayMode(.inline)
        .kmeansInfo(export, topic: .choosingK)
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("How many groups?").font(.headline)
            Text("k-means needs to be told k up front. Fit it once per k and "
                 + "watch the leftover error (inertia) drop — steeply at first, "
                 + "then levelling off once the clusters match real structure. "
                 + "That bend is the elbow.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var kPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("k").font(.caption)
                Spacer()
                Text("\(selectedK) clusters").font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Picker("k", selection: $selectedK) {
                ForEach(export.elbow) { e in Text("\(e.k)").tag(e.k) }
            }
            .pickerStyle(.segmented)
        }
    }

    private var reading: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Reading the curve").font(.headline)
            Text("From 1 to 4 clusters, inertia falls hard — each extra centroid "
                 + "is capturing a real group. From 4 onward it barely moves: "
                 + "extra clusters are just splitting groups that were already "
                 + "well separated. The elbow sits at k = \(export.k), which is "
                 + "also how many segments this data was generated from.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// Inertia against k, with the currently picked k highlighted.
struct ElbowCurveView: View {
    let elbow: [ElbowPoint]
    let selectedK: Int

    var body: some View {
        Canvas { context, size in
            guard !elbow.isEmpty else { return }
            let pad: CGFloat = 24
            let ks = elbow.map { Double($0.k) }
            let values = elbow.map(\.inertia)
            let kLo = ks.min()!, kHi = ks.max()!
            let vLo = 0.0, vHi = values.max()!

            func sx(_ k: Double) -> CGFloat {
                pad + CGFloat((k - kLo) / max(kHi - kLo, 1e-9)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat((v - vLo) / max(vHi - vLo, 1e-9)) * (size.height - 2 * pad)
            }

            var line = Path()
            for (i, e) in elbow.enumerated() {
                let p = CGPoint(x: sx(Double(e.k)), y: sy(e.inertia))
                if i == 0 { line.move(to: p) } else { line.addLine(to: p) }
            }
            context.stroke(line, with: .color(.accentColor), lineWidth: 2)

            for e in elbow {
                let p = CGPoint(x: sx(Double(e.k)), y: sy(e.inertia))
                let picked = e.k == selectedK
                let r: CGFloat = picked ? 6 : 3.5
                context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(picked ? .primary : .accentColor))
                context.draw(Text("\(e.k)").font(.caption2).foregroundStyle(.secondary),
                             at: CGPoint(x: p.x, y: size.height - 6), anchor: .top)
            }
        }
    }
}
