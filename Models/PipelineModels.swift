//
//  PipelineModels.swift
//  VisualML
//
//  Data structures shared across the ML pipeline. As we add stages (TF-IDF,
//  LSA, classifier) their result types will live here too.
//

import Foundation

/// The hyperparameters (knobs) that control how raw text becomes a matrix.
/// Later stages (weighting, LSA `k`, classifier params) will add fields here.
struct PipelineConfig {
    /// Drop very common, low-signal words ("the", "of", ...).
    var removeStopwords: Bool = true
    /// Keep a term only if it appears in at least this many documents.
    var minDocFreq: Int = 2
    /// Keep at most this many terms (the most frequent ones win).
    var maxVocab: Int = 150
}

/// The Bag-of-Words **document-term matrix**.
///
/// - `vocab`: the columns — one entry per kept term.
/// - `counts[d][t]`: how many times term `vocab[t]` occurs in document `d`.
/// - `labels` / `categories`: the class of each row (for grouping & coloring).
struct DocTermMatrix {
    let vocab: [String]        // length V
    let counts: [[Double]]     // D rows × V columns
    let labels: [Int]          // length D
    let categories: [String]   // length D

    var documentCount: Int { counts.count }
    var vocabularySize: Int { vocab.count }

    /// The largest single cell value — used to scale heatmap brightness.
    var maxCount: Double { counts.flatMap { $0 }.max() ?? 1 }

    /// Total occurrences of each term across all documents (the column sums).
    var termTotals: [Double] {
        var totals = [Double](repeating: 0, count: vocab.count)
        for row in counts {
            for t in row.indices { totals[t] += row[t] }
        }
        return totals
    }
}
