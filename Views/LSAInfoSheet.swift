//
//  LSAInfoSheet.swift
//  VisualML
//
//  "About LSA" help sheet: what LSA does, how each point's coordinates are
//  computed, a live worked example from the real data, which component carries
//  the class signal, and how Raw vs TF-IDF changes the plot.
//

import SwiftUI
import Foundation

struct LSAInfoSheet: View {
    @Environment(\.dismiss) private var dismiss

    /// The current matrix + config, so the worked example uses live numbers.
    var matrix: DocTermMatrix?
    var config: PipelineConfig

    // Computed once on appear (each example runs two LSAs — don't redo per frame).
    @State private var example: LSAExample?

    var body: some View {
        NavigationStack {
            List {
                Section("What LSA does") {
                    entry("Latent Semantic Analysis = truncated SVD",
                          "It factorizes the document-term matrix into a few latent "
                          + "components — directions in word-space along which documents "
                          + "vary the most. Each component is a weighted blend of many "
                          + "words (a 'topic'), not a single word.")
                    entry("Why",
                          "It compresses each document from a V-dimensional word vector "
                          + "(here V ≈ \(matrix?.vocabularySize ?? 0)) down to just 2 "
                          + "numbers, keeping the structure that matters: co-occurring "
                          + "words merge, noise drops, similar documents end up near each other.")
                    entry("It's unsupervised",
                          "LSA never sees the class labels. Any clustering is the geometry "
                          + "of the words themselves — the colors are added afterwards, only to check.")
                }

                Section("How each point is computed") {
                    entry("1 · Gram matrix",
                          "G = W · Wᵀ   (a D×D matrix). Entry G[i][j] is the dot product of "
                          + "documents i and j — how much word-mass they share.")
                    entry("2 · Eigendecomposition",
                          "G = U · Λ · Uᵀ. The eigenvectors U are the latent directions; the "
                          + "eigenvalues λ say how dominant each is. Singular values σᵢ = √λᵢ.")
                    entry("3 · Coordinates",
                          "A document d's position on component c is\n\n"
                          + "    coord(d, c) = U[d, c] · σ_c\n\n"
                          + "For the 2-D plot we keep components 1 and 2:\n"
                          + "    (x, y) = ( U[d,1]·σ₁ ,  U[d,2]·σ₂ ).")
                }

                if let ex = example {
                    workedSection(ex)
                    separationSection(ex)
                    rawVsTfidfSection(ex)
                }

                Section("What the plot represents") {
                    entry("Each dot = one document", "All \(matrix?.documentCount ?? 0) "
                          + "documents are placed by their latent coordinates.")
                    entry("The axes = the top 2 components",
                          "Horizontal = component 1 (strongest variation), vertical = "
                          + "component 2. Each axis is a blend of words, not a single word.")
                    entry("Distance = similarity",
                          "Documents with a similar word-mix sit close together, so the two "
                          + "topics drift into separate clouds — without ever being told the labels.")
                    entry("The scree bars",
                          "Below the plot, the bars are σ₁, σ₂, … — how much each component "
                          + "captures. Tall leading bars mean 2-D is enough to see the structure.")
                }
            }
            .navigationTitle("About LSA")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear {
                if example == nil, let m = matrix {
                    example = LSAExample(matrix: m, config: config)
                }
            }
        }
    }

    // MARK: - Live sections

    private func workedSection(_ ex: LSAExample) -> some View {
        Section("Worked example (live)") {
            Text("Document #\(ex.docNumber) (\(ex.category.capitalized)) · input: \(ex.scheme), L2 \(ex.l2 ? "on" : "off").")
                .font(.caption).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 4) {
                mono("σ₁ = \(f(ex.sigma1))    σ₂ = \(f(ex.sigma2))")
                mono("x = U[1,1]·σ₁ = \(f4(ex.u1)) · \(f(ex.sigma1)) = \(f(ex.x))")
                mono("y = U[1,2]·σ₂ = \(f4(ex.u2)) · \(f(ex.sigma2)) = \(f(ex.y))")
                Text("→ document #\(ex.docNumber) is plotted at (\(f(ex.x)), \(f(ex.y))).")
                    .font(.caption).bold()
            }
            .padding(.vertical, 2)
        }
    }

    private func separationSection(_ ex: LSAExample) -> some View {
        Section("Which axis separates the classes (live)") {
            Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("class mean").gridColumnAlignment(.leading)
                    Text("comp 1 (x)"); Text("comp 2 (y)")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)
                GridRow {
                    Text(ex.class0Name.capitalized).gridColumnAlignment(.leading)
                    Text(f(ex.mean0c1)); Text(f(ex.mean0c2))
                }.font(.caption2)
                GridRow {
                    Text(ex.class1Name.capitalized).gridColumnAlignment(.leading)
                    Text(f(ex.mean1c1)); Text(f(ex.mean1c2))
                }.font(.caption2)
                GridRow {
                    Text("gap").gridColumnAlignment(.leading).bold()
                    Text(f(ex.gap1)).bold(); Text(f(ex.gap2)).bold()
                }.font(.caption2)
            }
            .padding(.vertical, 2)
            Text("Component 1 captures the variation both classes SHARE (common filler "
                 + "words) — tiny gap. Component 2 captures the topic — the big gap, so the "
                 + "clouds separate \(ex.gap2 >= ex.gap1 ? "vertically" : "horizontally").")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func rawVsTfidfSection(_ ex: LSAExample) -> some View {
        Section("Raw vs TF-IDF (live)") {
            Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Text("input matrix").gridColumnAlignment(.leading)
                    Text("class separation")
                }
                .font(.caption2).bold().foregroundStyle(.secondary)
                GridRow { Text("Raw counts").gridColumnAlignment(.leading); Text(f(ex.rawBestGap)) }.font(.caption2)
                GridRow { Text("TF-IDF").gridColumnAlignment(.leading); Text(f(ex.tfidfBestGap)) }.font(.caption2)
            }
            .padding(.vertical, 2)
            Text("Same documents, different cell values → different Gram matrix → "
                 + "different plot. TF-IDF dims the filler words every document shares, so "
                 + "the latent axes line up with topic and the clouds pull further apart.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - Helpers

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

/// Computes a real LSA worked example from a `DocTermMatrix`: it runs LSA under
/// both Raw and TF-IDF weighting (mirroring `Weighting` + `MatrixMath`) and pulls
/// out document #1's coordinates and the class-separation along each component.
struct LSAExample {
    let docNumber: Int
    let category: String
    let scheme: String
    let l2: Bool
    let sigma1: Double, sigma2: Double
    let x: Double, y: Double
    let u1: Double, u2: Double             // recovered eigenvector entries = coord / σ
    let mean0c1: Double, mean1c1: Double   // class means on component 1
    let mean0c2: Double, mean1c2: Double   // class means on component 2
    let rawBestGap: Double, tfidfBestGap: Double
    let class0Name: String, class1Name: String

    var gap1: Double { abs(mean0c1 - mean1c1) }
    var gap2: Double { abs(mean0c2 - mean1c2) }

    init?(matrix m: DocTermMatrix, config: PipelineConfig) {
        guard m.documentCount > 1, m.vocabularySize > 0 else { return nil }
        let weighting = Weighting()
        let mathEngine = MatrixMath()
        let rawLSA = mathEngine.performLSA(
            weighting.weighted(m, scheme: .rawCounts, l2normalize: config.l2normalize), components: 2)
        let tfLSA = mathEngine.performLSA(
            weighting.weighted(m, scheme: .tfidf, l2normalize: config.l2normalize), components: 2)
        guard (rawLSA.coords.first?.count ?? 0) >= 2,
              (tfLSA.coords.first?.count ?? 0) >= 2 else { return nil }

        let active = (config.weighting == .tfidf) ? tfLSA : rawLSA

        // Mean of a class's coordinate on one component.
        func classMean(_ r: SVDResult, _ comp: Int, _ label: Int) -> Double {
            var sum = 0.0, n = 0
            for i in r.labels.indices where r.labels[i] == label { sum += r.coords[i][comp]; n += 1 }
            return n > 0 ? sum / Double(n) : 0
        }
        // The larger class-mean gap across the two components = "separating power".
        func bestGap(_ r: SVDResult) -> Double {
            max(abs(classMean(r, 0, 0) - classMean(r, 0, 1)),
                abs(classMean(r, 1, 0) - classMean(r, 1, 1)))
        }

        let d = 0
        let s1 = active.singularValues[0], s2 = active.singularValues[1]
        let xx = active.coords[d][0], yy = active.coords[d][1]

        self.docNumber = d + 1
        self.category = m.categories[d]
        self.scheme = config.weighting.rawValue
        self.l2 = config.l2normalize
        self.sigma1 = s1; self.sigma2 = s2
        self.x = xx; self.y = yy
        self.u1 = s1 != 0 ? xx / s1 : 0
        self.u2 = s2 != 0 ? yy / s2 : 0
        self.mean0c1 = classMean(active, 0, 0); self.mean1c1 = classMean(active, 0, 1)
        self.mean0c2 = classMean(active, 1, 0); self.mean1c2 = classMean(active, 1, 1)
        self.rawBestGap = bestGap(rawLSA)
        self.tfidfBestGap = bestGap(tfLSA)

        // Human-readable class names by label index.
        var names = ["class 0", "class 1"]
        for i in m.labels.indices { let l = m.labels[i]; if l == 0 || l == 1 { names[l] = m.categories[i] } }
        self.class0Name = names[0]; self.class1Name = names[1]
    }
}
