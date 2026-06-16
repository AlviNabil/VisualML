//
//  MatrixInfoSheet.swift
//  VisualML
//
//  The "What am I looking at?" help sheet for the matrix screen. Presented from
//  an ⓘ info button — the standard iOS pattern for inline documentation.
//

import SwiftUI

struct MatrixInfoSheet: View {
    // `dismiss` is provided by the environment; calling it closes the sheet.
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("The matrix") {
                    entry("Rows = documents",
                          "Every row is one document (one news paragraph). Rows are "
                          + "grouped by class, so all Business documents form the top "
                          + "block and all Sport documents the bottom block, with a "
                          + "divider line between them.")
                    entry("Columns = terms",
                          "Every column is one word from the vocabulary — the set of "
                          + "words kept after lowercasing, removing stop-words, and "
                          + "dropping very rare words. Columns are in alphabetical order.")
                    entry("A cell = a word's value in a document",
                          "Brighter means a higher value: a higher count in Raw mode, "
                          + "or a higher TF-IDF weight in TF-IDF mode. Empty (white) "
                          + "cells mean the word does not appear in that document.")
                }

                Section("TF-IDF") {
                    entry("Term Frequency × Inverse Document Frequency",
                          "A way to score words by how *informative* they are, not just "
                          + "how often they appear.")
                    entry("TF — term frequency",
                          "How many times a word appears in a document. Common within a "
                          + "topic, but easy to be misled by filler words.")
                    entry("IDF — inverse document frequency",
                          "Down-weights words that appear in many documents and "
                          + "up-weights rare ones:\n\nidf(t) = ln((1 + D) / (1 + df(t))) + 1\n\n"
                          + "where D is the number of documents and df(t) is how many "
                          + "documents contain the word t.")
                    entry("Why it matters",
                          "Multiplying TF × IDF dims words that are everywhere "
                          + "('everyone', 'the') and brightens rare, telling words — so "
                          + "the columns that actually separate Sport from Business stand out.")
                }

                Section("L2 normalization") {
                    entry("Make every document the same 'length'",
                          "Rescales each document's row so the square root of its summed "
                          + "squared values equals 1 (unit length).")
                    entry("Why it matters",
                          "Without it, a longer document looks 'louder' simply because it "
                          + "has more words. After L2 normalization, documents are compared "
                          + "on equal footing — only the *mix* of words matters, not the "
                          + "document's length.")
                }
            }
            .navigationTitle("What am I looking at?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    /// One titled paragraph inside a section.
    private func entry(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline).bold()
            Text(body).font(.caption).foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    MatrixInfoSheet()
}
