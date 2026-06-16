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
