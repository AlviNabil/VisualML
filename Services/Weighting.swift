//
//  Weighting.swift
//  VisualML
//
//  Re-weights the Bag-of-Words matrix. Raw counts over-value words that appear
//  everywhere ("the", "everyone"); TF-IDF fixes that by multiplying each count
//  by the term's inverse document frequency.
//

import Foundation

/// Turns a `DocTermMatrix` (integer counts) into a `WeightedMatrix` (real weights).
struct Weighting {

    /// Apply the chosen scheme (and optional L2 normalization).
    ///
    /// The math:
    ///   df(t)   = number of documents containing term t
    ///   idf(t)  = ln( (1 + D) / (1 + df(t)) ) + 1        // smoothed, never 0
    ///   W[d,t]  = count(d,t) · idf(t)                     // tf · idf
    ///   (optional) divide each row by its L2 norm ‖W_d‖₂
    func weighted(_ m: DocTermMatrix,
                  scheme: WeightingScheme,
                  l2normalize: Bool) -> WeightedMatrix {
        let docCount = m.documentCount
        let vocabCount = m.vocabularySize

        // df(t): in how many documents does term t occur? (column non-zero count)
        var df = [Double](repeating: 0, count: vocabCount)
        for row in m.counts {
            for t in 0..<vocabCount where row[t] > 0 { df[t] += 1 }
        }

        // idf(t): smoothed inverse document frequency. For raw counts we leave
        // idf = 1 (multiplying by 1 changes nothing), so the same code path works.
        var idf = [Double](repeating: 1, count: vocabCount)
        if scheme == .tfidf {
            for t in 0..<vocabCount {
                idf[t] = log((1.0 + Double(docCount)) / (1.0 + df[t])) + 1.0
            }
        }

        // W[d][t] = tf(d,t) · idf(t). tf is just the raw count.
        var rows = m.counts
        if scheme == .tfidf {
            for d in 0..<docCount {
                for t in 0..<vocabCount { rows[d][t] = m.counts[d][t] * idf[t] }
            }
        }

        // Optional L2 normalization: rescale each document vector to length 1, so
        // a long document doesn't look "brighter" just for having more words.
        if l2normalize {
            for d in 0..<docCount {
                let norm = sqrt(rows[d].reduce(0) { $0 + $1 * $1 })
                if norm > 0 {
                    for t in 0..<vocabCount { rows[d][t] /= norm }
                }
            }
        }

        return WeightedMatrix(vocab: m.vocab, rows: rows, idf: idf,
                              labels: m.labels, categories: m.categories)
    }
}
