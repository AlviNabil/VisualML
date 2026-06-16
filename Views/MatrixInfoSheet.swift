//
//  MatrixInfoSheet.swift
//  VisualML
//
//  The "What am I looking at?" help sheet for the matrix screen. Presented from
//  an ⓘ info button — the standard iOS pattern for inline documentation.
//

import SwiftUI
import Foundation

struct MatrixInfoSheet: View {
    // `dismiss` is provided by the environment; calling it closes the sheet.
    @Environment(\.dismiss) private var dismiss

    /// Passed in so the "Worked example" section can show REAL numbers from the
    /// document the user is actually looking at. Optional: if nil we skip it.
    var matrix: DocTermMatrix? = nil

    var body: some View {
        NavigationStack {
            List {
                Section("The matrix") {
                    entry("Rows = documents",
                          "Every row is one document (one news paragraph). Rows are "
                          + "grouped by class, so all Business documents form the top "
                          + "block and all Sport documents the bottom block, with a "
                          + "divider line between them.")
                    entry("Columns = terms",
                          "Every column is one word from the vocabulary — the set of "
                          + "words kept after lowercasing, removing stop-words, and "
                          + "dropping very rare words. Columns are in alphabetical order.")
                    entry("A cell = a word's value in a document",
                          "Brighter means a higher value: a higher count in Raw mode, "
                          + "or a higher TF-IDF weight in TF-IDF mode. Empty (white) "
                          + "cells mean the word does not appear in that document.")
                }

                Section("TF-IDF") {
                    entry("Term Frequency × Inverse Document Frequency",
                          "A way to score words by how *informative* they are, not just "
                          + "how often they appear.")
                    entry("TF — term frequency",
                          "How many times a word appears in a document.")
                    entry("IDF — inverse document frequency",
                          "Down-weights words in many documents, up-weights rare ones:\n\n"
                          + "idf(t) = ln((1 + D) / (1 + df(t))) + 1\n\n"
                          + "where D is the number of documents and df(t) is how many "
                          + "documents contain the word t.")
                    entry("Why it matters",
                          "TF × IDF dims words that are everywhere ('everyone', 'the') "
                          + "and brightens rare, telling words — so the columns that "
                          + "actually separate Sport from Business stand out.")
                }

                // Live worked example using the real matrix the user is viewing.
                if let m = matrix, let ex = WorkedExample(matrix: m) {
                    tfidfExampleSection(ex)
                    l2ExampleSection(ex)
                }

                Section("L2 normalization") {
                    entry("Make every document the same 'length'",
                          "Rescales each document's row so the square root of its summed "
                          + "squared values equals 1 (unit length).")
                    entry("Why it matters",
                          "Without it, a longer document looks 'louder' simply because it "
                          + "has more words. After L2 normalization, documents are compared "
                          + "on equal footing — only the *mix* of words matters, not the "
                          + "document's length.")
                }
            }
            .navigationTitle("What am I looking at?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    // MARK: - Worked-example sections

    private func tfidfExampleSection(_ ex: WorkedExample) -> some View {
        Section("Worked example: TF-IDF") {
            Text("Document #\(ex.docNumber) (\(ex.category.capitalized)), with D = \(ex.docCount) documents. A few of its terms:")
                .font(.caption).foregroundStyle(.secondary)

            Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 6) {
                GridRow {
                    Text("term").gridColumnAlignment(.leading)
                    Text("tf"); Text("df"); Text("idf"); Text("tf·idf")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)

                ForEach(ex.terms, id: \.term) { t in
                    GridRow {
                        Text(t.term).gridColumnAlignment(.leading).bold()
                        Text("\(Int(t.tf))")
                        Text("\(t.df)")
                        Text(String(format: "%.2f", t.idf))
                        Text(String(format: "%.2f", t.tfidf))
                    }
                    .font(.caption2)
                }
            }
            .padding(.vertical, 2)

            if let h = ex.terms.first {
                Text("e.g. idf(\(h.term)) = ln((1+\(ex.docCount)) / (1+\(h.df))) + 1 = \(String(format: "%.2f", h.idf)),\n"
                     + "so tf·idf = \(Int(h.tf)) × \(String(format: "%.2f", h.idf)) = \(String(format: "%.2f", h.tfidf)).")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func l2ExampleSection(_ ex: WorkedExample) -> some View {
        Section("Worked example: L2 normalization") {
            Text("Divide every weight by the row's length ‖w‖₂ = √(Σ (tf·idf)²) = \(String(format: "%.2f", ex.l2norm)). "
                 + "Afterwards the row's squared values sum to 1.")
                .font(.caption).foregroundStyle(.secondary)

            Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 6) {
                GridRow {
                    Text("term").gridColumnAlignment(.leading)
                    Text("tf·idf"); Text("÷ ‖w‖₂")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)

                ForEach(ex.terms, id: \.term) { t in
                    GridRow {
                        Text(t.term).gridColumnAlignment(.leading).bold()
                        Text(String(format: "%.2f", t.tfidf))
                        Text(String(format: "%.3f", t.normalized))
                    }
                    .font(.caption2)
                }
            }
            .padding(.vertical, 2)
        }
    }

    /// One titled paragraph inside a section.
    private func entry(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline).bold()
            Text(body).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

/// Computes a small, real TF-IDF + L2 worked example from a `DocTermMatrix`,
/// for one document and a handful of illustrative terms. Mirrors `Weighting`.
struct WorkedExample {
    struct Term {
        let term: String
        let tf: Double
        let df: Int
        let idf: Double
        let tfidf: Double
        let normalized: Double
    }

    let docNumber: Int          // 1-indexed, for display
    let category: String
    let docCount: Int
    let l2norm: Double
    let terms: [Term]

    init?(matrix m: DocTermMatrix, docIndex: Int = 0) {
        let docCount = m.documentCount
        let vocabCount = m.vocabularySize
        guard docCount > 0, vocabCount > 0, docIndex < docCount else { return nil }

        // df(t): documents containing term t (column non-zero count).
        var df = [Int](repeating: 0, count: vocabCount)
        for row in m.counts {
            for t in 0..<vocabCount where row[t] > 0 { df[t] += 1 }
        }

        // idf and the full tf·idf row for the chosen document.
        let idf = (0..<vocabCount).map { log((1.0 + Double(docCount)) / (1.0 + Double(df[$0]))) + 1.0 }
        let counts = m.counts[docIndex]
        let tfidf = (0..<vocabCount).map { counts[$0] * idf[$0] }
        let l2 = sqrt(tfidf.reduce(0) { $0 + $1 * $1 })

        // Choose terms that make the contrast clear: the 3 highest-weight terms
        // PLUS the 2 most common (lowest-idf) terms present in this document.
        let present = (0..<vocabCount).filter { counts[$0] > 0 }
        guard !present.isEmpty else { return nil }
        var pick = Array(present.sorted { tfidf[$0] > tfidf[$1] }.prefix(3))
        for t in present.sorted(by: { idf[$0] < idf[$1] }).prefix(2) where !pick.contains(t) {
            pick.append(t)
        }
        pick.sort { tfidf[$0] > tfidf[$1] }

        self.docNumber = docIndex + 1
        self.category = m.categories[docIndex]
        self.docCount = docCount
        self.l2norm = l2
        self.terms = pick.map { t in
            Term(term: m.vocab[t], tf: counts[t], df: df[t], idf: idf[t],
                 tfidf: tfidf[t], normalized: l2 > 0 ? tfidf[t] / l2 : 0)
        }
    }
}

#Preview {
    MatrixInfoSheet()
}
