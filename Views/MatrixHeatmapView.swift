//
//  MatrixHeatmapView.swift
//  VisualML
//
//  Renders a document-term matrix (raw or weighted) as a heatmap, plus the
//  Bag-of-Words / TF-IDF stage screen.
//

import SwiftUI

/// Draws a D × V matrix of real values as a grid of colored cells using `Canvas`.
///
/// Why `Canvas`? The matrix can have ~100 × ~120 ≈ 12,000 cells. Making that many
/// SwiftUI views would be painfully slow. `Canvas` paints them all as ONE view.
///
/// It draws plain arrays (`rows`, `labels`, `maxValue`) instead of a specific
/// type, so the SAME view renders both raw counts and TF-IDF weights.
/// - Rows = documents, grouped by class (so the block structure is obvious).
/// - Columns = terms (alphabetical).
/// - Cell color = the document's class; brightness ∝ the cell's value.
struct MatrixHeatmapView: View {
    let rows: [[Double]]   // D × V values
    let labels: [Int]      // class label per row
    let maxValue: Double   // largest cell value, for brightness scaling

    init(rows: [[Double]], labels: [Int], maxValue: Double) {
        self.rows = rows
        self.labels = labels
        self.maxValue = max(maxValue, 0.0001)   // guard against divide-by-zero
    }

    /// Convenience: draw a weighted matrix directly.
    init(weighted m: WeightedMatrix) {
        self.init(rows: m.rows, labels: m.labels, maxValue: m.maxValue)
    }

    /// Row display order: documents sorted by class label, so all of class 0
    /// sit together, then all of class 1. That makes the two blocks visible.
    private var rowOrder: [Int] {
        labels.indices.sorted { labels[$0] < labels[$1] }
    }

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    var body: some View {
        Canvas { context, size in
            let docCount = rows.count
            guard docCount > 0, let vocabCount = rows.first?.count, vocabCount > 0 else { return }

            let cellW = size.width / CGFloat(vocabCount)
            let cellH = size.height / CGFloat(docCount)
            // Leave a small vertical gap between rows so each document reads as
            // its own band instead of one congested block. We shrink the drawn
            // cell height and center it, leaving rowGap/2 of empty space above
            // and below — the row pitch (cellH) is unchanged.
            let rowGap = min(cellH * 0.4, 2.5)
            let drawH = max(cellH - rowGap, 0.5)

            var previousLabel: Int? = nil
            for (displayRow, docIndex) in rowOrder.enumerated() {
                let label = labels[docIndex]
                let base = color(forLabel: label)
                let row = rows[docIndex]
                let y = CGFloat(displayRow) * cellH

                // Paint each non-zero cell. Empty cells stay transparent.
                for t in 0..<vocabCount where row[t] > 0 {
                    // Brightness ∝ value relative to the largest cell. The 0.30
                    // floor keeps the smallest non-zero cells visible; clamp at 1.
                    let intensity = min(0.30 + 0.70 * (row[t] / maxValue), 1.0)
                    let rect = CGRect(x: CGFloat(t) * cellW, y: y + rowGap / 2,
                                      width: max(cellW, 0.5), height: drawH)
                    context.fill(Path(rect), with: .color(base.opacity(intensity)))
                }

                // Thin line where the class changes (the block boundary).
                if let prev = previousLabel, prev != label {
                    var line = Path()
                    line.move(to: CGPoint(x: 0, y: y))
                    line.addLine(to: CGPoint(x: size.width, y: y))
                    context.stroke(line, with: .color(.primary.opacity(0.6)), lineWidth: 1)
                }
                previousLabel = label
            }
        }
    }
}

/// The matrix stage screen: explanation, the Raw ⇄ TF-IDF toggle, the heatmap,
/// and the most-influential terms (which reorder when you switch weighting).
struct BagOfWordsView: View {
    // @ObservedObject: this view watches a VM that someone ELSE owns
    // (ContentView owns it via @StateObject). We just observe & react.
    @ObservedObject var viewModel: PipelineViewModel

    // Drives the help sheet, opened from the ⓘ button.
    @State private var showInfo = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let weighted = viewModel.weightedMatrix {
                    caption(weighted)
                    controls
                    MatrixHeatmapView(weighted: weighted)
                        .frame(height: 440)
                        .padding(8)
                        .background(Color.gray.opacity(0.10))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    legend(weighted)
                    topTerms(weighted)

                    // Move to the next pipeline stage: LSA (2-D projection).
                    NavigationLink {
                        LSAView(viewModel: viewModel)
                    } label: {
                        Label("View LSA projection", systemImage: "chart.dots.scatter")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 8)
                } else {
                    ProgressView("Building matrix…")
                        .frame(maxWidth: .infinity, minHeight: 220)
                }
            }
            .padding()
        }
        .navigationTitle("Bag of Words")
//        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // ⓘ — the standard iOS spot for "what am I looking at?" documentation.
            ToolbarItem(placement: .topBarTrailing) {
                Button { showInfo = true } label: {
                    Image(systemName: "info.circle")
                }
                .accessibilityLabel("About this screen")
            }
        }
        // A sheet is the iOS-standard container for this kind of help. Detents let
        // the user open it half-height and drag up for the full text.
        .sheet(isPresented: $showInfo) {
            // Pass the real matrix so the sheet's worked example uses live numbers.
            MatrixInfoSheet(matrix: viewModel.matrix)
                .presentationDetents([.medium, .large])
        }
        // Build it the first time we arrive (cheap; rebuilt later when knobs change).
        .onAppear {
            if viewModel.matrix == nil {
                Task{
                    await viewModel.buildBagOfWords()
                }
            }
        }
    }

    // MARK: - Controls (the knobs)

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            // A segmented control bound STRAIGHT to the config. Flipping it mutates
            // config.weighting (a @Published value) → the VM publishes → this body
            // re-runs → viewModel.weightedMatrix recomputes → the heatmap redraws.
            Picker("Weighting", selection: $viewModel.config.weighting) {
                ForEach(WeightingScheme.allCases) { scheme in
                    Text(scheme.rawValue).tag(scheme)
                }
            }
            .pickerStyle(.segmented)

            Toggle("L2-normalize each document", isOn: $viewModel.config.l2normalize)
                .font(.subheadline)
        }
    }

    // MARK: - Pieces

    private func caption(_ m: WeightedMatrix) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Document-Term Matrix").font(.headline)
            Text("\(m.documentCount) documents × \(m.vocabularySize) terms. "
                 + (viewModel.config.weighting == .tfidf
                    ? "Cells are TF-IDF weights (count × idf): ubiquitous words dim, rare telling words brighten."
                    : "Cells are raw counts; brighter = higher count.")
                 + " Rows grouped by class.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func legend(_ m: WeightedMatrix) -> some View {
        HStack(spacing: 16) {
            ForEach(classLegend(m), id: \.label) { item in
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color(forLabel: item.label))
                        .frame(width: 12, height: 12)
                    Text(item.name.capitalized).font(.caption)
                }
            }
        }
    }

    /// The most influential terms, ranked by total weight (column sums). In raw
    /// mode that's the most *frequent* words; in TF-IDF mode the list reorders to
    /// the most *discriminative* words — the visible payoff of weighting.
    private func topTerms(_ m: WeightedMatrix) -> some View {
        let totals = m.termWeightTotals
        let ranked = Array(m.vocab.indices.sorted { totals[$0] > totals[$1] }.prefix(15))
        let isTFIDF = viewModel.config.weighting == .tfidf
        return VStack(alignment: .leading, spacing: 8) {
            Text(isTFIDF ? "Top terms by total TF-IDF weight" : "Most frequent terms")
                .font(.headline)
            // FlowLayout sizes each chip to its own word and wraps to the next
            // line, so a long word like "everyone" is never split across lines.
            FlowLayout(spacing: 8) {
                ForEach(ranked, id: \.self) { i in
                    HStack(spacing: 4) {
                        Text(m.vocab[i]).font(.caption).bold()
                        Text(isTFIDF ? String(format: "%.1f", totals[i]) : "\(Int(totals[i]))")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    .lineLimit(1)
                    .fixedSize()                 // take the word's full width; never wrap
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Color.gray.opacity(0.15))
                    .clipShape(Capsule())
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Helpers

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    /// Distinct (name, label) pairs for the legend, sorted by label.
    private func classLegend(_ m: WeightedMatrix) -> [(name: String, label: Int)] {
        var seen: [Int: String] = [:]
        for (label, name) in zip(m.labels, m.categories) where seen[label] == nil {
            seen[label] = name
        }
        return seen.map { (name: $0.value, label: $0.key) }.sorted { $0.label < $1.label }
    }
}

//#Preview {
//    let previewViewModel = PipelineViewModel()
//    
//    return BagOfWordsView(viewModel: previewViewModel)
//        // Attach an async task to run the moment the preview canvas renders
//        .task {
//            // 1. Load the real CSV data, exactly like ContentView does
//            await previewViewModel.loadDataset()
//            
//            // 2. Force the matrix to build immediately after the data is ready
//            await previewViewModel.buildBagOfWords()
//        }
//}
