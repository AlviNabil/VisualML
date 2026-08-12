//
//  KMeansInfoSheet.swift
//  VisualML
//
//  The maths behind the k-means stage, worked through with the numbers this
//  app shipped: why clustering has no target to fit, the assign/update loop
//  and why it always improves, k-means++ and why the start matters, and how
//  to read the elbow curve.
//

import SwiftUI

struct KMeansInfoSheet: View {
    let export: ClusterExport
    var initialTopic: Topic = .setup

    @Environment(\.dismiss) private var dismiss
    @State private var topic: Topic?

    enum Topic: String, CaseIterable, Identifiable {
        case setup = "Setup"
        case choosingK = "Choosing k"
        case algorithm = "Algorithm"
        case result = "Result"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Topic", selection: Binding(
                    get: { topic ?? initialTopic },
                    set: { topic = $0 })) {
                    ForEach(Topic.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding()

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        switch topic ?? initialTopic {
                        case .setup: setupSection
                        case .choosingK: choosingKSection
                        case .algorithm: algorithmSection
                        case .result: resultSection
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 28)
                }
            }
            .navigationTitle("The maths")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    // MARK: - Setup

    private var setupSection: some View {
        let d = export.dataset
        return VStack(alignment: .leading, spacing: 14) {
            heading("No target to fit")
            body("Every earlier stage minimised the gap between a prediction "
                 + "and a known answer. Clustering has no known answer: each of "
                 + "the \(d.n) customers is just a point, "
                 + "\(d.axisLabels.joined(separator: " and ").lowercased()). "
                 + "The task is to find groups in the points themselves.")

            heading("What k-means optimises instead")
            body("Given a guess at k group centres, inertia measures how tightly "
                 + "the points cling to their nearest one:")
            formula("inertia = Σᵢ ‖xᵢ − centroid(xᵢ)‖²")
            body("Small inertia means every point sits close to the centre it "
                 + "was assigned to — the centres are a good summary of the data.")

            heading("Why the features are rescaled first")
            body(String(format: "%@ ranges over about %.0f units here; %@ ranges "
                        + "over about %.0f. Left alone, distance would be almost "
                        + "entirely decided by the larger-range feature.",
                        d.axisLabels[0], d.maxs[0] - d.mins[0],
                        d.axisLabels[1], d.maxs[1] - d.mins[1]))
            formula("x_scaled = (x − mean) / std")
            note("Every distance in this stage is computed on the rescaled "
                 + "features. The centroids you see on the map are converted "
                 + "back to the original units afterwards, so they still read "
                 + "as real spend and real visit counts.")
        }
    }

    // MARK: - Choosing k

    private var choosingKSection: some View {
        let elbow = export.elbow
        let jump = elbow.count > 1 ? elbow[0].inertia - elbow[safe: elbow.count - 1]!.inertia : 0
        return VStack(alignment: .leading, spacing: 14) {
            heading("Inertia only ever falls")
            body("Adding a centroid can never make inertia worse — in the "
                 + "extreme, k equal to the number of points lets every point be "
                 + "its own centroid, at zero inertia. So the lowest inertia is "
                 + "never the right way to choose k; it would always pick the "
                 + "largest k available.")
            formula("k = 1  →  inertia = " + fmt(elbow.first?.inertia ?? 0)
                    + "\nk = " + String(export.dataset.n) + "  →  inertia = 0  (not useful)")

            heading("What the elbow actually measures")
            body("Instead, look at how much each additional cluster buys you. "
                 + "While centroids are still merging genuinely separate groups, "
                 + "inertia drops fast. Once every real group already has its own "
                 + "centroid, an extra one can only split a group that didn't "
                 + "need splitting — and the curve goes flat.")
            elbowTable

            heading("Reading this dataset's curve")
            body(String(format: "Total drop from k=1 to k=%d: %.0f. Almost all "
                        + "of that happens by k=%d — after that, doubling k barely "
                        + "moves the curve. That flattening point is the elbow.",
                        elbow.last?.k ?? 0, jump, export.k))
        }
    }

    private var elbowTable: some View {
        VStack(spacing: 3) {
            ForEach(export.elbow) { e in
                elbowRow(e)
            }
        }
        .padding(10)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func elbowRow(_ e: ElbowPoint) -> some View {
        HStack {
            Text("k = \(e.k)").font(.caption.monospacedDigit()).frame(width: 50, alignment: .leading)
            Text(fmt(e.inertia)).font(.caption.monospacedDigit())
            Spacer()
            if e.k == export.k {
                Text("← elbow").font(.caption2).foregroundStyle(Color.accentColor)
            }
        }
    }

    // MARK: - Algorithm

    private var algorithmSection: some View {
        let best = export.restarts.min { $0.finalInertia < $1.finalInertia }
        let worst = export.restarts.max { $0.finalInertia < $1.finalInertia }
        return VStack(alignment: .leading, spacing: 14) {
            heading("Starting the centroids: k-means++")
            body("Placing the first k centroids purely at random tends to bunch "
                 + "several of them inside the same real group, leaving another "
                 + "group with none. k-means++ instead picks the first centroid "
                 + "uniformly, then each following one with probability "
                 + "proportional to its squared distance from the nearest centroid "
                 + "already chosen — points far from existing centroids are far "
                 + "more likely to be picked next.")
            formula("P(point i chosen next) ∝ min distance² to an existing centroid")

            heading("Then two steps, repeated")
            formula("assign:  each point → its nearest centroid\n"
                    + "update:  each centroid → the mean of its assigned points")
            body("Both steps can only lower or hold inertia — assigning to the "
                 + "nearest centroid can't increase anyone's distance, and the "
                 + "mean is the point that minimises total squared distance to a "
                 + "fixed group. So the loop always settles; the recorded rounds "
                 + "stop once no centroid moves.")

            heading("Why the starting point matters")
            if let best, let worst, best.seed != worst.seed {
                body(String(format: "Seed %d and seed %d fit the exact same data "
                            + "and the exact same k, and still reached different "
                            + "answers: inertia %.2f against %.2f.",
                            best.seed, worst.seed, best.finalInertia, worst.finalInertia))
                note("k-means always converges — but only to *a* local optimum of "
                     + "inertia, not provably the best one. That is why real "
                     + "usage runs several random starts and keeps whichever "
                     + "reaches the lowest inertia, exactly like the five starts "
                     + "on the previous step.")
            }
        }
    }

    // MARK: - Result

    private var resultSection: some View {
        let best = export.best
        return VStack(alignment: .leading, spacing: 14) {
            heading("The centroids are the model")
            body("A fitted regression is fully described by its weights. A "
                 + "fitted k-means is fully described by its centroids — "
                 + String(export.k) + " points, each one a typical member of its "
                 + "group. Nothing else about the training data needs to be kept "
                 + "to classify a new customer: just find the nearest centroid.")

            heading("Checking against the truth")
            body(String(format: "This data was generated from %d known segments, "
                        + "so the discovered clusters can be checked against "
                        + "them: %.0f%% of customers landed in the cluster that "
                        + "matches their true segment.", export.k, best.purity * 100))
            note("This check only works because the data is simulated. On real "
                 + "customers there is no ground truth to compare against — "
                 + "inertia and the elbow curve are the only signals available.")

            heading("What k-means assumes")
            bullet("Clusters are roughly round and similar in size — a long, "
                   + "thin, or oddly shaped group gets cut apart by nearest-centroid "
                   + "assignment.")
            bullet("k is chosen correctly. Too few clusters merges real groups; "
                   + "too many splits a real one in half.")
            bullet("Distance in the rescaled features is a meaningful notion of "
                   + "similarity for this problem.")
        }
    }

    // MARK: - Building blocks

    private func heading(_ text: String) -> some View {
        Text(text).font(.headline).padding(.top, 4)
    }

    private func body(_ text: String) -> some View {
        Text(text).font(.subheadline).foregroundStyle(.secondary)
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").font(.subheadline).foregroundStyle(.secondary)
            Text(text).font(.subheadline).foregroundStyle(.secondary)
        }
    }

    private func formula(_ text: String) -> some View {
        Text(text)
            .font(.callout.monospaced())
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func note(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "lightbulb.fill")
                .font(.caption2).foregroundStyle(.yellow)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.yellow.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func fmt(_ value: Double) -> String {
        value >= 100 ? String(format: "%.0f", value) : String(format: "%.2f", value)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Reaching it from any step

struct KMeansInfoButton: ViewModifier {
    let export: ClusterExport
    let topic: KMeansInfoSheet.Topic
    @State private var showing = false

    func body(content: Content) -> some View {
        content
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showing = true } label: { Image(systemName: "info.circle") }
                        .accessibilityLabel("The maths behind this step")
                }
            }
            .sheet(isPresented: $showing) {
                KMeansInfoSheet(export: export, initialTopic: topic)
                    .presentationDetents([.large])
            }
    }
}

extension View {
    /// Attaches the k-means maths sheet, opening on the section for this step.
    func kmeansInfo(_ export: ClusterExport, topic: KMeansInfoSheet.Topic) -> some View {
        modifier(KMeansInfoButton(export: export, topic: topic))
    }
}
