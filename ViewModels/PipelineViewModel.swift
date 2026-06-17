//
//  PipelineViewModel.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import Foundation
import Combine

@MainActor
class PipelineViewModel: ObservableObject {
    @Published var dataPoints: [DataPoint] = []
    @Published var statusMessage: String = "Ready to load data"

    /// The hyperparameter knobs (so far: stop-words, minDocFreq, maxVocab).
    @Published var config = PipelineConfig()
    /// The Bag-of-Words matrix once built (nil until we build it).
    @Published var matrix: DocTermMatrix?

    private let loader = DatasetLoader()
    private let nlp = NLPProcessor()
    private let weighting = Weighting()
    private let math = MatrixMath()
    private let classifier = Classifier()

    /// Classifier hyperparameters (kind, η, iterations, λ). Changing any of these
    /// retrains the model — see `trainedModel`.
    @Published var classifierParams = ClassifierParams()

    /// The matrix after applying the current weighting + normalization knobs.
    ///
    /// This is a *computed* property, not stored: it re-derives from `matrix` and
    /// `config` every time it's read. Because the View observes this VM, flipping
    /// the weighting toggle changes `config` (a @Published value) → the View's
    /// body re-runs → this recomputes → the heatmap redraws. That's the
    /// "real-time" effect, with no manual refresh code.
    var weightedMatrix: WeightedMatrix? {
        guard let matrix else { return nil }
        return weighting.weighted(matrix,
                                  scheme: config.weighting,
                                  l2normalize: config.l2normalize)
    }

    /// LSA of the ACTIVE weighting (drives the LSA page). Stored + cached, and
    /// computed OFF the main actor by `computeLSA()`, so the UI never blocks.
    @Published private(set) var lsaResult: SVDResult?
    /// LSA of the OTHER weighting — used only for the in-app Raw-vs-TF-IDF comparison.
    @Published private(set) var lsaComparison: SVDResult?
    @Published private(set) var isComputingLSA = false

    private var lsaCache: [String: SVDResult] = [:]

    /// A cache key capturing everything that changes an LSA result.
    private func lsaKey(_ scheme: WeightingScheme) -> String {
        "\(scheme.rawValue)|l2:\(config.l2normalize)|sw:\(config.removeStopwords)"
        + "|mdf:\(config.minDocFreq)|mv:\(config.maxVocab)|n:\(dataPoints.count)"
    }

    /// Compute LSA for the active weighting (the page) and the other weighting
    /// (the in-app comparison), off the main actor and cached. Called on appear.
    func computeLSA() async {
        guard let m = matrix else { return }
        let active = config.weighting
        let other: WeightingScheme = (active == .tfidf) ? .rawCounts : .tfidf
        isComputingLSA = true
        lsaResult = await cachedLSA(active, m)
        lsaComparison = await cachedLSA(other, m)
        isComputingLSA = false
    }

    private func cachedLSA(_ scheme: WeightingScheme, _ m: DocTermMatrix) async -> SVDResult {
        let key = lsaKey(scheme)
        if let hit = lsaCache[key] { return hit }   // already computed → instant
        let w = weighting.weighted(m, scheme: scheme, l2normalize: config.l2normalize)
        let engine = math
        // `Task.detached` runs the heavy eigendecomposition on a background thread;
        // `.value` hops the result back to the main actor.
        let result = await Task.detached(priority: .userInitiated) {
            engine.performLSA(w, components: 2)
        }.value
        lsaCache[key] = result
        return result
    }

    /// The trained classifier, fit on the current LSA points with the current
    /// params. Computed (cheap: 2-D, ~100 points), so dragging a knob retrains and
    /// the boundary moves in real time — no manual refresh.
    var trainedModel: TrainedModel? {
        guard let lsa = lsaResult, lsa.documentCount > 0 else { return nil }
        return classifier.train(coords: lsa.coords, labels: lsa.labels,
                                categories: lsa.categories, params: classifierParams)
    }

    /// Run a typed sentence through the ENTIRE pipeline and predict its class:
    /// tokenize → count over the trained vocabulary → weight (idf, L2) exactly as
    /// in training → project onto the latent components → apply the boundary.
    func classify(_ text: String) -> Prediction? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let m = matrix, let lsa = lsaResult,
              let model = trainedModel, let wm = weightedMatrix else { return nil }

        // 1. Raw counts over the SAME vocabulary the model was trained on.
        let colOf = Dictionary(uniqueKeysWithValues: m.vocab.enumerated().map { ($1, $0) })
        var vec = [Double](repeating: 0, count: m.vocab.count)
        var matched = 0
        for token in nlp.tokenize(trimmed) {
            if let c = colOf[token] { vec[c] += 1; matched += 1 }
        }

        // 2. Weight it just like training: TF-IDF uses the trained idf, then L2.
        if config.weighting == .tfidf { for t in vec.indices { vec[t] *= wm.idf[t] } }
        if config.l2normalize {
            let norm = (vec.reduce(0) { $0 + $1 * $1 }).squareRoot()
            if norm > 0 { for t in vec.indices { vec[t] /= norm } }
        }

        // 3. Project onto the latent components:  coord_c = Σ_t vec[t] · V[t][c]
        //    (V = the term loadings; this is exactly W·V used for training docs).
        var x = 0.0, y = 0.0
        for t in 0..<m.vocab.count {
            x += vec[t] * lsa.termLoadings[t][0]
            y += vec[t] * lsa.termLoadings[t][1]
        }

        // 4. Apply the trained boundary (z ≥ 0 → class 1, for all three kinds).
        let z = model.w0 * x + model.w1 * y + model.b
        let label = z >= 0 ? 1 : 0
        let name = label == 1 ? model.class1 : model.class0
        let prob: Double? = (model.kind == .logistic)
            ? (label == 1 ? 1 / (1 + exp(-z)) : 1 - 1 / (1 + exp(-z)))
            : nil
        return Prediction(category: name, label: label, score: z,
                          probability: prob, x: x, y: y, matchedWords: matched)
    }

    func loadDataset() async{
        statusMessage = "Loading..."
        do {
            dataPoints.removeAll()
            // "dataset" matches dataset.csv in the app bundle.
            dataPoints = try loader.loadCSV(filename: "dataset")
//            dump(dataPoints.prefix(10))
            statusMessage = "Loaded \(dataPoints.count) documents."
        } catch {
            // `localizedDescription` comes from our DatasetError.errorDescription,
            // so this tells us exactly what went wrong.
            statusMessage = "Error: \(error.localizedDescription)"
        }
    }

    /// Build the Bag-of-Words document-term matrix from the loaded documents,
    /// using the current `config`. Fast enough (D≈120) to run on the main actor;
    /// we'll move heavier stages (SVD) off the main thread later.
    func buildBagOfWords() async {
        try? await Task.sleep(for: .seconds(2))
        guard !dataPoints.isEmpty else { return }
        matrix = nlp.buildMatrix(from: dataPoints, config: config)
    }

    /// How many documents belong to each class, sorted by label (0, 1, ...).
    /// Used by the UI to show a per-class count and legend.
    ///
    /// `Dictionary(grouping:by:)` buckets the documents by category, e.g.
    /// ["sport": [..60 items..], "business": [..60 items..]]. We then turn each
    /// bucket into a small tuple the View can display.
    var classCounts: [(name: String, label: Int, count: Int)] {
        Dictionary(grouping: dataPoints, by: { $0.category })
            .map { (name, items) in (name: name, label: items[0].label, count: items.count) }
            .sorted { $0.label < $1.label }
    }
}
