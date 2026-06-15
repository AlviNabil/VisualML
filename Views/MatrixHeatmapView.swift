//
//  MatrixHeatmapView.swift
//  VisualML
//
//  Renders the document-term matrix as a heatmap, plus the Bag-of-Words screen.
//

import SwiftUI

/// Draws the document-term matrix as a grid of colored cells using `Canvas`.
///
/// Why `Canvas`? The matrix can have ~120 × ~150 = 18,000 cells. Making 18,000
/// SwiftUI views would be painfully slow. `Canvas` draws them all as one view,
/// like painting on a single sheet — fast and smooth.
///
/// - Rows = documents, grouped by class (so the block structure is obvious).
/// - Columns = terms (alphabetical).
/// - Cell color = the document's class; brightness = the word count in that cell.
struct MatrixHeatmapView: View {
    let matrix: DocTermMatrix

    /// Row display order: documents sorted by class label, so all of class 0
    /// sit together, then all of class 1. That makes the two blocks visible.
    private var rowOrder: [Int] {
        matrix.labels.indices.sorted { matrix.labels[$0] < matrix.labels[$1] }
    }

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    var body: some View {
        Canvas { context, size in
            let docCount = matrix.documentCount
            let vocabCount = matrix.vocabularySize
            guard docCount > 0, vocabCount > 0 else { return }

            let cellW = size.width / CGFloat(vocabCount)
            let cellH = size.height / CGFloat(docCount)
            let maxC = max(matrix.maxCount, 1)

            var previousLabel: Int? = nil
            for (displayRow, docIndex) in rowOrder.enumerated() {
                let label = matrix.labels[docIndex]
                let base = color(forLabel: label)
                let row = matrix.counts[docIndex]
                let y = CGFloat(displayRow) * cellH

                // Paint each non-zero cell. Empty cells stay transparent.
                for t in 0..<vocabCount where row[t] > 0 {
                    let intensity = 0.25 + 0.75 * (row[t] / maxC)   // 0.25...1.0
                    let rect = CGRect(x: CGFloat(t) * cellW, y: y,
                                      width: max(cellW, 0.5), height: max(cellH, 0.5))
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

/// The Bag-of-Words stage screen: explanation + heatmap + most-frequent terms.
struct BagOfWordsView: View {
    // @ObservedObject: this view watches a VM that someone ELSE owns
    // (ContentView owns it via @StateObject). We just observe & react.
    @ObservedObject var viewModel: PipelineViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let matrix = viewModel.matrix {
                    caption(matrix)
                    MatrixHeatmapView(matrix: matrix)
                        .frame(height: 340)
                        .padding(8)
                        .background(Color.gray.opacity(0.10))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    legend(matrix)
                    topTerms(matrix)
                } else {
                    ProgressView("Building matrix…")
                        .frame(maxWidth: .infinity, minHeight: 220)
                }
            }
            .padding()
        }
        .navigationTitle("Bag of Words")
        .navigationBarTitleDisplayMode(.inline)
        // Build it the first time we arrive (cheap; rebuilt later when knobs change).
        .onAppear {
            if viewModel.matrix == nil { viewModel.buildBagOfWords() }
        }
    }

    // MARK: - Pieces

    private func caption(_ matrix: DocTermMatrix) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Document-Term Matrix")
                .font(.headline)
            Text("\(matrix.documentCount) documents × \(matrix.vocabularySize) terms. "
                 + "Each row is a document, each column a word; brighter = higher count. "
                 + "Rows are grouped by class.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func legend(_ matrix: DocTermMatrix) -> some View {
        HStack(spacing: 16) {
            ForEach(classLegend(matrix), id: \.label) { item in
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color(forLabel: item.label))
                        .frame(width: 12, height: 12)
                    Text(item.name.capitalized).font(.caption)
                }
            }
        }
    }

    /// The most frequent terms overall, shown as little chips.
    private func topTerms(_ matrix: DocTermMatrix) -> some View {
        let totals = matrix.termTotals
        let ranked = Array(matrix.vocab.indices.sorted { totals[$0] > totals[$1] }.prefix(15))
        return VStack(alignment: .leading, spacing: 8) {
            Text("Most frequent terms").font(.headline)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 84), spacing: 8)],
                      alignment: .leading, spacing: 8) {
                ForEach(ranked, id: \.self) { i in
                    HStack(spacing: 4) {
                        Text(matrix.vocab[i]).font(.caption).bold()
                        Text("\(Int(totals[i]))")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Color.gray.opacity(0.15))
                    .clipShape(Capsule())
                }
            }
        }
    }

    // MARK: - Helpers

    private func color(forLabel label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }

    /// Distinct (name, label) pairs for the legend, sorted by label.
    private func classLegend(_ matrix: DocTermMatrix) -> [(name: String, label: Int)] {
        var seen: [Int: String] = [:]
        for (label, name) in zip(matrix.labels, matrix.categories) where seen[label] == nil {
            seen[label] = name
        }
        return seen.map { (name: $0.value, label: $0.key) }.sorted { $0.label < $1.label }
    }
}
