//
//  NLPProcessor.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/13/26.
//

import Foundation
import NaturalLanguage

/// Turns raw text into numbers: tokenization + the Bag-of-Words matrix.
///
/// This is the first place where "two distinct datasets" become a single
/// numerical object we can do linear algebra on.
class NLPProcessor {

    /// Very common words that carry little topic signal. Removing them (a
    /// hyperparameter) lets the matrix focus on meaningful, discriminative terms.
    private let stopwords: Set<String> = [
        "the","a","an","and","or","but","if","of","to","in","on","for","with",
        "as","at","by","from","into","over","after","before","that","this",
        "these","those","it","its","is","are","was","were","be","been","being",
        "his","her","their","they","he","she","we","you","not","no","so",
        "than","then","there","here","up","down","out","off","again","more",
        "most","some","such","only","own","same","too","very","can","will",
        "just","also","had","has","have","do","does","did","while","who","whom",
        "which","what","when","where","why","how","all","any","both","each",
        "few","other","about","against","between","through","during"
    ]

    /// Split text into lowercase word tokens.
    ///
    /// Uses Apple's `NLTokenizer` for proper word boundaries, then:
    ///   - takes the part before any apostrophe ("company's" -> "company"),
    ///   - keeps only purely-alphabetic tokens of length >= 2
    ///     (drops numbers, punctuation, and single letters).
    func tokenize(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text

        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let raw = text[range].lowercased()
            // "company's" -> "company"; "didn't" -> "didn" (rare, acceptable).
            let word = raw.split(separator: "'").first.map(String.init) ?? raw
            if word.count >= 2 && word.allSatisfy({ $0.isLetter }) {
                tokens.append(word)
            }
            return true   // keep scanning
        }
        return tokens
    }

    /// Build the document-term matrix from documents, honoring the config knobs.
    ///
    /// Steps (this is the Bag-of-Words algorithm):
    ///   1. Tokenize every document.
    ///   2. Count, per term: document-frequency (in how many docs) and total
    ///      frequency (overall).
    ///   3. Choose the vocabulary (the columns): drop stopwords, drop terms below
    ///      `minDocFreq`, then keep the `maxVocab` most frequent of what remains.
    ///   4. Fill the D × V matrix with counts.
    func buildMatrix(from documents: [DataPoint], config: PipelineConfig) -> DocTermMatrix {
        // 1. Tokenize once, reuse below.
        let tokenized: [[String]] = documents.map { tokenize($0.text) }

        // 2. Document frequency + total frequency for every term we saw.
        var docFreq: [String: Int] = [:]
        var totalFreq: [String: Int] = [:]
        for tokens in tokenized {
            var seenInThisDoc = Set<String>()
            for token in tokens {
                totalFreq[token, default: 0] += 1
                // `insert` returns inserted == true only the first time per doc,
                // so docFreq counts DOCUMENTS, not occurrences.
                if seenInThisDoc.insert(token).inserted {
                    docFreq[token, default: 0] += 1
                }
            }
        }

        // 3. Filter, then rank by total frequency and keep the top `maxVocab`.
        var candidates = totalFreq.keys.filter { term in
            if config.removeStopwords && stopwords.contains(term) { return false }
            return (docFreq[term] ?? 0) >= config.minDocFreq
        }
        candidates.sort { a, b in
            let fa = totalFreq[a] ?? 0, fb = totalFreq[b] ?? 0
            return fa != fb ? fa > fb : a < b   // freq desc, then alphabetical
        }
        // Take the most frequent, then display alphabetically for stable columns.
        let vocab = Array(candidates.prefix(config.maxVocab)).sorted()
        let columnOf = Dictionary(uniqueKeysWithValues: vocab.enumerated().map { ($1, $0) })

        // 4. Fill the matrix.
        var counts = [[Double]](
            repeating: [Double](repeating: 0, count: vocab.count),
            count: documents.count
        )
        for (d, tokens) in tokenized.enumerated() {
            for token in tokens {
                if let c = columnOf[token] { counts[d][c] += 1 }
            }
        }

        return DocTermMatrix(
            vocab: vocab,
            counts: counts,
            labels: documents.map { $0.label },
            categories: documents.map { $0.category }
        )
    }
}
