//
//  NLPProcessor.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/13/26.
//

import Foundation
import NaturalLanguage

class NLPProcessor {
    // Maps each unique word to a specific index in our vector
    private(set) var vocabulary: [String: Int] = [:]
    
    /// Step 1: Scan all data to build the distinct vocabulary
    func buildVocabulary(from data: [DataPoint]) {
        let tokenizer = NLTokenizer(unit: .word)
        var uniqueWords = Set<String>()
        
        for point in data {
            tokenizer.string = point.text
            
            tokenizer.enumerateTokens(in: point.text.startIndex..<point.text.endIndex) { tokenRange, _ in
                // Lowercase to ensure "The" and "the" are treated as the same feature
                let word = String(point.text[tokenRange]).lowercased()
                uniqueWords.insert(word)
                return true // Continue scanning
            }
        }
        
        // Assign a fixed index to each word
        for (index, word) in uniqueWords.enumerated() {
            vocabulary[word] = index
        }
        print("Vocabulary built with \(vocabulary.count) unique words.")
    }
    
    /// Step 2: Convert a single text string into a numerical array (BoW vector)
    func generateBagOfWords(for text: String) -> [Double] {
        // Initialize an array of zeros matching the vocabulary size
        var vector = Array(repeating: 0.0, count: vocabulary.count)
        
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            let word = String(text[tokenRange]).lowercased()
            
            // If the word exists in our vocabulary, increment its count
            if let index = vocabulary[word] {
                vector[index] += 1.0
            }
            return true
        }
        
        return vector
    }
}
