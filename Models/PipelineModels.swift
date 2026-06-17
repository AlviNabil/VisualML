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

    /// How matrix cells are valued: raw counts, or TF-IDF weights.
    var weighting: WeightingScheme = .rawCounts
    /// If true, scale each document row to unit length (so long docs don't dominate).
    var l2normalize: Bool = false
}

/// The two ways we value a cell. `RawValue` is the label shown in the UI Picker.
enum WeightingScheme: String, CaseIterable, Identifiable {
    case rawCounts = "Raw counts"
    case tfidf = "TF-IDF"
    var id: String { rawValue }
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

/// A *weighted* document-term matrix (e.g. TF-IDF). Same shape as
/// `DocTermMatrix`, but cells are real-valued weights instead of integer counts.
struct WeightedMatrix {
    let vocab: [String]        // length V
    let rows: [[Double]]       // D × V weights
    let idf: [Double]          // length V — the idf used for each term (all 1 for raw)
    let labels: [Int]          // length D
    let categories: [String]   // length D

    var documentCount: Int { rows.count }
    var vocabularySize: Int { vocab.count }

    /// The largest single weight — used to scale heatmap brightness.
    var maxValue: Double { rows.flatMap { $0 }.max() ?? 1 }

    /// Total weight of each term across all documents (column sums). In raw mode
    /// this equals the count totals; in TF-IDF mode it ranks the most *influential*
    /// terms, which is different — that's the whole point.
    var termWeightTotals: [Double] {
        var totals = [Double](repeating: 0, count: vocab.count)
        for row in rows {
            for t in row.indices { totals[t] += row[t] }
        }
        return totals
    }
}

/// Result of Latent Semantic Analysis (truncated SVD).
///
/// LSA factorizes the weighted matrix W = U·Σ·Vᵀ and keeps the top components.
/// `coords` are the document embeddings U·Σ — each document's position in the
/// new low-dimensional "latent topic" space; for the scatter plot we use k = 2.
struct SVDResult {
    let coords: [[Double]]        // D × k  — document coordinates (rows of U·Σ)
    let singularValues: [Double]  // length k — σ for the kept components
    let spectrum: [Double]        // all singular values, descending (for the scree plot)
    let termLoadings: [[Double]]  // V × k  — how strongly each term loads on each component
    let vocab: [String]           // length V — column labels (for the dominant-words table)
    let labels: [Int]             // length D
    let categories: [String]      // length D

    var documentCount: Int { coords.count }
}
