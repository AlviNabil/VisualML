//
//  LSAView.swift
//  VisualML
//
//  Latent Semantic Analysis stage: every document drawn as a point in the 2-D
//  latent space (the top two SVD components), colored by class, plus a scree
//  plot of the singular values.
//

import SwiftUI

/// Draws documents as points in 2-D using `Canvas`. Auto-scales the coordinates
/// to fill the available space, and flips the y-axis (screen y grows downward).
struct ScatterPlotView: View {
    let coords: [[Double]]   // D × 2 (or more; we use the first two columns)
    let labels: [Int]

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    var body: some View {
        Canvas { context, size in
            guard !coords.isEmpty, coords[0].count >= 2 else { return }

            // Bounds of the data, so we can map it into the view rectangle.
            let xs = coords.map { $0[0] }
            let ys = coords.map { $0[1] }
            let minX = xs.min()!, maxX = xs.max()!
            let minY = ys.min()!, maxY = ys.max()!
            let pad: CGFloat = 18
            let plotW = size.width - 2 * pad
            let plotH = size.height - 2 * pad

            // Map a data value to a screen coordinate.
            func sx(_ v: Double) -> CGFloat {
                let range = maxX - minX
                let t = range > 0 ? (v - minX) / range : 0.5
                return pad + CGFloat(t) * plotW
            }
            func sy(_ v: Double) -> CGFloat {
                let range = maxY - minY
                let t = range > 0 ? (v - minY) / range : 0.5
                return pad + (1 - CGFloat(t)) * plotH   // flip: data up = screen up
            }

            // Faint center axes (where each latent component = 0).
            var axes = Path()
            axes.move(to: CGPoint(x: sx(0), y: pad));        axes.addLine(to: CGPoint(x: sx(0), y: size.height - pad))
            axes.move(to: CGPoint(x: pad, y: sy(0)));        axes.addLine(to: CGPoint(x: size.width - pad, y: sy(0)))
            context.stroke(axes, with: .color(.gray.opacity(0.25)), lineWidth: 1)

            // One dot per document.
            let r: CGFloat = 4
            for d in 0..<coords.count {
                let p = CGPoint(x: sx(coords[d][0]), y: sy(coords[d][1]))
                let rect = CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)
                context.fill(Path(ellipseIn: rect),
                             with: .color(color(forLabel: labels[d]).opacity(0.85)))
            }
        }
    }
}

/// The LSA stage screen.
struct LSAView: View {
    @ObservedObject var viewModel: PipelineViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let lsa = viewModel.lsaResult, lsa.documentCount > 0 {
                    caption
                    ScatterPlotView(coords: lsa.coords, labels: lsa.labels)
                        .frame(height: 360)
                        .padding(8)
                        .background(Color.gray.opacity(0.10))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    legend(lsa)
                    scree(lsa)
                } else {
                    ProgressView("Computing LSA…")
                        .frame(maxWidth: .infinity, minHeight: 220)
                }
            }
            .padding()
        }
        .navigationTitle("LSA (2-D)")
    }

    // MARK: - Pieces

    private var caption: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Documents in latent space").font(.headline)
            Text("Each dot is a document, projected onto the top 2 SVD components of "
                 + "the current matrix (\(viewModel.config.weighting.rawValue)). Nearby dots "
                 + "are topically similar — the two classes should form separate clouds.")
                .font(.caption).foregroundStyle(.secondary)
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

    /// Scree plot: a bar per singular value. The first two (the plot's axes) are
    /// highlighted; their height vs the rest shows how much these 2-D capture.
    private func scree(_ lsa: SVDResult) -> some View {
        let top = Array(lsa.spectrum.prefix(12))
        let maxV = top.max() ?? 1
        return VStack(alignment: .leading, spacing: 8) {
            Text("Singular values (scree)").font(.headline)
            Text("How much each latent component matters. Taller bar = more variance captured.")
                .font(.caption).foregroundStyle(.secondary)
            Canvas { context, size in
                let n = top.count
                guard n > 0 else { return }
                let gap: CGFloat = 4
                let bw = (size.width - gap * CGFloat(n - 1)) / CGFloat(n)
                for i in 0..<n {
                    let bh = maxV > 0 ? CGFloat(top[i] / maxV) * size.height : 0
                    let rect = CGRect(x: CGFloat(i) * (bw + gap), y: size.height - bh,
                                      width: bw, height: bh)
                    let isAxis = i < 2
                    context.fill(Path(roundedRect: rect, cornerRadius: 2),
                                 with: .color((isAxis ? Color.green : Color.gray)
                                    .opacity(isAxis ? 0.85 : 0.4)))
                }
            }
            .frame(height: 80)
        }
    }

    // MARK: - Helpers

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
