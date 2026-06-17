//
//  LSAInfoSheet.swift
//  VisualML
//
//  "About LSA" help sheet. Reads the already-computed (cached) LSA results from
//  the view model — it does NOT run any LSA itself, so it opens instantly.
//

import SwiftUI

struct LSAInfoSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var viewModel: PipelineViewModel

    private var active: SVDResult? { viewModel.lsaResult }
    private var config: PipelineConfig { viewModel.config }

    var body: some View {
        NavigationStack {
            List {
                whatSection
                howSection

                if let r = active, r.documentCount > 0, r.singularValues.count >= 2 {
                    workedSection(r)
                    documentTableSection(r)
                    dominantWordsSection(r)
                    separationSection(r)
                    if let other = viewModel.lsaComparison {
                        rawVsTfidfSection(active: r, other: other)
                    }
                } else {
                    Section { ProgressView("Computing…").frame(maxWidth: .infinity) }
                }

                plotSection
            }
            .navigationTitle("About LSA")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }

    // MARK: - Static explanation

    private var whatSection: some View {
        Section("What LSA does") {
            entry("Latent Semantic Analysis = truncated SVD",
                  "It factorizes the document-term matrix into a few latent components — "
                  + "directions in word-space along which documents vary the most. Each "
                  + "component is a weighted blend of many words (a 'topic'), not one word.")
            entry("Why",
                  "It compresses each document from a V-dimensional word vector (here "
                  + "V ≈ \(viewModel.matrix?.vocabularySize ?? 0)) down to just 2 numbers, "
                  + "keeping the structure that matters: co-occurring words merge, noise drops.")
            entry("It's unsupervised",
                  "LSA never sees the class labels. Any clustering is the geometry of the "
                  + "words themselves — colors are added afterwards, only to check.")
        }
    }

    private var howSection: some View {
        Section("How each point is computed") {
            entry("1 · Gram matrix",
                  "G = W · Wᵀ (D×D). G[i][j] is the dot product of documents i and j — "
                  + "how much word-mass they share.")
            entry("2 · Eigendecomposition",
                  "G = U · Λ · Uᵀ. Eigenvectors U are the latent directions; eigenvalues λ "
                  + "say how dominant each is. Singular values σᵢ = √λᵢ.")
            entry("3 · Coordinates",
                  "coord(d, c) = U[d, c] · σ_c.   For the 2-D plot we keep components 1 & 2:\n"
                  + "(x, y) = ( U[d,1]·σ₁ ,  U[d,2]·σ₂ ).")
        }
    }

    private var plotSection: some View {
        Section("What the plot represents") {
            entry("Each dot = one document", "All \(active?.documentCount ?? 0) documents, placed by their latent coordinates.")
            entry("The axes = the top 2 components",
                  "Horizontal = component 1 (strongest variation), vertical = component 2. "
                  + "Each axis is a blend of words (see the dominant-words table).")
            entry("Distance = similarity",
                  "Documents with a similar word-mix sit close, so the two topics drift into "
                  + "separate clouds — without ever being told the labels.")
        }
    }

    // MARK: - Live sections

    private func workedSection(_ r: SVDResult) -> some View {
        let s1 = r.singularValues[0], s2 = r.singularValues[1]
        let x = r.coords[0][0], y = r.coords[0][1]
        return Section("Worked example (live)") {
            Text("Document #1 (\(r.categories[0].capitalized)) · input: \(config.weighting.rawValue), L2 \(config.l2normalize ? "on" : "off").")
                .font(.caption).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                mono("σ₁ = \(f(s1))    σ₂ = \(f(s2))")
                mono("x = U[1,1]·σ₁ = \(f4(s1 != 0 ? x/s1 : 0)) · \(f(s1)) = \(f(x))")
                mono("y = U[1,2]·σ₂ = \(f4(s2 != 0 ? y/s2 : 0)) · \(f(s2)) = \(f(y))")
                Text("→ document #1 is plotted at (\(f(x)), \(f(y))).").font(.caption).bold()
            }
            .padding(.vertical, 2)
        }
    }

    private func documentTableSection(_ r: SVDResult) -> some View {
        Section("Document coordinates — first 10 (live)") {
            Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 6) {
                GridRow {
                    Text("doc").gridColumnAlignment(.leading)
                    Text("class").gridColumnAlignment(.leading)
                    Text("comp 1"); Text("comp 2")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)
                ForEach(0..<min(10, r.documentCount), id: \.self) { d in
                    GridRow {
                        Text("#\(d + 1)").gridColumnAlignment(.leading)
                        Text(r.categories[d].capitalized).gridColumnAlignment(.leading)
                        Text(f(r.coords[d][0])); Text(f(r.coords[d][1]))
                    }
                    .font(.caption2)
                }
            }
            .padding(.vertical, 2)
        }
    }

    private func dominantWordsSection(_ r: SVDResult) -> some View {
        Section("Dominant words per component — top 10 (live)") {
            Text("The words with the largest loading on each axis. The sign shows which side "
                 + "of the axis they pull toward (e.g. one class vs the other on component 2).")
                .font(.caption).foregroundStyle(.secondary)
            componentWords("Component 1 (x)", r, 0)
            componentWords("Component 2 (y)", r, 1)
        }
    }

    private func componentWords(_ title: String, _ r: SVDResult, _ comp: Int) -> some View {
        let ranked = r.vocab.indices
            .sorted { abs(r.termLoadings[$0][comp]) > abs(r.termLoadings[$1][comp]) }
            .prefix(10)
        return VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).bold()
            Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 4) {
                ForEach(Array(ranked), id: \.self) { t in
                    GridRow {
                        Text(r.vocab[t]).gridColumnAlignment(.leading)
                        Text(f(r.termLoadings[t][comp]))
                    }
                    .font(.caption2)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func separationSection(_ r: SVDResult) -> some View {
        let n0 = className(r, 0), n1 = className(r, 1)
        let g1 = abs(classMean(r, 0, 0) - classMean(r, 0, 1))
        let g2 = abs(classMean(r, 1, 0) - classMean(r, 1, 1))
        return Section("Which axis separates the classes (live)") {
            Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("class mean").gridColumnAlignment(.leading)
                    Text("comp 1 (x)"); Text("comp 2 (y)")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)
                GridRow {
                    Text(n0.capitalized).gridColumnAlignment(.leading)
                    Text(f(classMean(r, 0, 0))); Text(f(classMean(r, 1, 0)))
                }.font(.caption2)
                GridRow {
                    Text(n1.capitalized).gridColumnAlignment(.leading)
                    Text(f(classMean(r, 0, 1))); Text(f(classMean(r, 1, 1)))
                }.font(.caption2)
                GridRow {
                    Text("gap").gridColumnAlignment(.leading).bold()
                    Text(f(g1)).bold(); Text(f(g2)).bold()
                }.font(.caption2)
            }
            .padding(.vertical, 2)
            Text("Component 1 captures the variation both classes SHARE (filler words) — "
                 + "tiny gap. Component 2 captures the topic — the big gap, so the clouds "
                 + "separate \(g2 >= g1 ? "vertically" : "horizontally").")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func rawVsTfidfSection(active r: SVDResult, other: SVDResult) -> some View {
        let raw = config.weighting == .rawCounts ? r : other
        let tf = config.weighting == .tfidf ? r : other
        return Section("Raw vs TF-IDF (live)") {
            Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("input matrix").gridColumnAlignment(.leading)
                    Text("class separation")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)
                GridRow { Text("Raw counts").gridColumnAlignment(.leading); Text(f(bestGap(raw))) }.font(.caption2)
                GridRow { Text("TF-IDF").gridColumnAlignment(.leading); Text(f(bestGap(tf))) }.font(.caption2)
            }
            .padding(.vertical, 2)
            Text("Same documents, different cell values → different Gram matrix → different "
                 + "plot. TF-IDF dims the filler words every document shares, so the latent "
                 + "axes line up with topic and the clouds pull further apart.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - Math helpers

    private func classMean(_ r: SVDResult, _ comp: Int, _ label: Int) -> Double {
        var sum = 0.0, n = 0
        for i in r.labels.indices where r.labels[i] == label { sum += r.coords[i][comp]; n += 1 }
        return n > 0 ? sum / Double(n) : 0
    }
    private func bestGap(_ r: SVDResult) -> Double {
        max(abs(classMean(r, 0, 0) - classMean(r, 0, 1)),
            abs(classMean(r, 1, 0) - classMean(r, 1, 1)))
    }
    private func className(_ r: SVDResult, _ label: Int) -> String {
        for i in r.labels.indices where r.labels[i] == label { return r.categories[i] }
        return "class \(label)"
    }

    // MARK: - View helpers

    private func entry(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline).bold()
            Text(body).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
    private func mono(_ s: String) -> some View { Text(s).font(.caption2.monospaced()) }
    private func f(_ v: Double) -> String { String(format: "%.2f", v) }
    private func f4(_ v: Double) -> String { String(format: "%.4f", v) }
}
