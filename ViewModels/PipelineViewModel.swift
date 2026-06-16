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

    /// 2-D LSA projection of the current weighted matrix. Computed (not stored),
    /// so it follows whatever weighting/normalization the user has chosen — the
    /// downstream stage recomputes automatically. Small enough (D≈100) to run
    /// inline; heavier datasets would move this off the main actor.
    var lsaResult: SVDResult? {
        guard let w = weightedMatrix else { return nil }
        return math.performLSA(w, components: 2)
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
