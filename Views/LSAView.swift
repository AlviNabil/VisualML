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
    /// Optional decision boundary  w0·x + w1·y + b = 0  to overlay.
    var boundary: (w0: Double, w1: Double, b: Double)? = nil
    /// Also draw the z = ±1 margins (used for SVM).
    var showMargins: Bool = false
    /// Optional extra point to spotlight (a typed sentence's position).
    var highlight: (x: Double, y: Double, label: Int)? = nil

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    var body: some View {
        Canvas { context, size in
            guard !coords.isEmpty, coords[0].count >= 2 else { return }

            // Bounds of the data, so we can map it into the view rectangle.
            // Include the highlight point so a typed sentence is never off-screen.
            var xs = coords.map { $0[0] }
            var ys = coords.map { $0[1] }
            if let h = highlight { xs.append(h.x); ys.append(h.y) }
            let minX = xs.min()!, maxX = xs.max()!
            let minY = ys.min()!, maxY = ys.max()!
            let pad: CGFloat = 18
            let plotW = size.width - 2 * pad
            let plotH = size.height - 2 * pad

            // EQUAL-ASPECT (isotropic) mapping: both axes share ONE scale, so the
            // plot shows the embedding's true geometry. Scaling each axis
            // independently to fill the box (the naive approach) would magnify a
            // low-variance axis into fake "scatter" — exactly what L2-normalization's
            // collapsed length-axis would look like — so we deliberately avoid it.
            let rangeX = max(maxX - minX, 1e-9)
            let rangeY = max(maxY - minY, 1e-9)
            let centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2
            let scale = min(plotW / rangeX, plotH / rangeY) * 0.95   // one scale for x & y
            let midX = pad + plotW / 2, midY = pad + plotH / 2

            // Map a data value to a screen coordinate (y flipped: data up = screen up).
            func sx(_ v: Double) -> CGFloat { midX + CGFloat((v - centerX) * scale) }
            func sy(_ v: Double) -> CGFloat { midY - CGFloat((v - centerY) * scale) }

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

            // Decision boundary  a0·x + a1·y + bb = 0  (and optional ±1 margins).
            // Solve for whichever variable keeps the line well-defined, then map to screen.
            func drawLine(_ a0: Double, _ a1: Double, _ bb: Double, dashed: Bool) {
                var line = Path()
                if abs(a1) >= abs(a0) {
                    guard abs(a1) > 1e-12 else { return }
                    line.move(to: CGPoint(x: sx(minX), y: sy(-(a0 * minX + bb) / a1)))
                    line.addLine(to: CGPoint(x: sx(maxX), y: sy(-(a0 * maxX + bb) / a1)))
                } else {
                    line.move(to: CGPoint(x: sx(-(a1 * minY + bb) / a0), y: sy(minY)))
                    line.addLine(to: CGPoint(x: sx(-(a1 * maxY + bb) / a0), y: sy(maxY)))
                }
                context.stroke(line, with: .color(dashed ? .gray : .primary),
                               style: StrokeStyle(lineWidth: dashed ? 1 : 2,
                                                  dash: dashed ? [5, 4] : []))
            }
            if let bd = boundary {
                if showMargins {
                    drawLine(bd.w0, bd.w1, bd.b - 1, dashed: true)
                    drawLine(bd.w0, bd.w1, bd.b + 1, dashed: true)
                }
                drawLine(bd.w0, bd.w1, bd.b, dashed: false)
            }

            // Spotlight a typed sentence: a filled dot (predicted color) + ring.
            if let h = highlight {
                let p = CGPoint(x: sx(h.x), y: sy(h.y))
                context.fill(Path(ellipseIn: CGRect(x: p.x - 6, y: p.y - 6, width: 12, height: 12)),
                             with: .color(color(forLabel: h.label)))
                context.stroke(Path(ellipseIn: CGRect(x: p.x - 9, y: p.y - 9, width: 18, height: 18)),
                               with: .color(.primary), lineWidth: 2.5)
            }
        }
    }
}

/// The LSA stage screen.
struct LSAView: View {
    @ObservedObject var viewModel: PipelineViewModel
    @State private var showInfo = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let lsa = viewModel.lsaResult, lsa.documentCount > 0 {
                    caption
                    configChips                 // what config this page was opened with
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
        .toolbar {
            // ⓘ — opens the math + a live worked example.
            ToolbarItem(placement: .topBarTrailing) {
                Button { showInfo = true } label: { Image(systemName: "info.circle") }
                    .accessibilityLabel("About LSA")
            }
        }
        .sheet(isPresented: $showInfo) {
            LSAInfoSheet(viewModel: viewModel)
                .presentationDetents([.medium, .large])
        }
        // Compute LSA off the main actor when the screen appears (cached after).
        .task { await viewModel.computeLSA() }
    }

    // MARK: - Pieces

    private var caption: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Documents in latent space").font(.headline)
            Text("Each dot is a document, projected onto the top 2 SVD components of "
                 + "the matrix below. Nearby dots are topically similar — the two classes "
                 + "should form separate clouds. Tap ⓘ for the math and a worked example.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    /// Chips showing the weighting + L2 state this page was navigated with.
    private var configChips: some View {
        HStack(spacing: 8) {
            chip("Input: \(viewModel.config.weighting.rawValue)", .blue)
            chip("L2: \(viewModel.config.l2normalize ? "on" : "off")",
                 viewModel.config.l2normalize ? .green : .gray)
        }
    }

    private func chip(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2).bold()
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
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
