//
//  DatasetLoader.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import Foundation

/// Errors the loader can throw. Conforming to `LocalizedError` means
/// `error.localizedDescription` gives us a human-readable message we can
/// show in the UI instead of a cryptic code.
enum DatasetError: Error, LocalizedError {
    case fileNotFound(String)
    case missingColumns
    case empty

    var errorDescription: String? {
        switch self {
        case .fileNotFound(let name):
            return "Could not find \(name).csv inside the app bundle."
        case .missingColumns:
            return "The CSV must contain both a 'label' and a 'text' column."
        case .empty:
            return "The dataset has no usable rows."
        }
    }
}

/// Loads a bundled CSV of labeled text into an array of `DataPoint`.
///
/// We parse the CSV by hand instead of using Apple's `CreateML` /`MLDataTable`,
/// for two reasons:
///   1. CreateML is a *macOS-only* framework — importing it breaks the iOS build.
///   2. Doing it ourselves keeps the whole pipeline transparent, which is the
///      entire point of this teaching app.
class DatasetLoader {

    /// Reads `<filename>.csv` from the app bundle and returns the parsed rows.
    /// - Parameter filename: the resource name WITHOUT the `.csv` extension.
    func loadCSV(filename: String) throws -> [DataPoint] {
        // `Bundle.main` is the running app's container. `url(forResource:...)`
        // finds a file we added to the "Copy Bundle Resources" build phase.
        guard let url = Bundle.main.url(forResource: filename, withExtension: "csv") else {
            throw DatasetError.fileNotFound(filename)
        }

        // Read the whole file into one big String.
        let raw = try String(contentsOf: url, encoding: .utf8)

        // Normalize Windows line endings (\r\n) to \n, then split into lines.
        // `omittingEmptySubsequences` drops blank lines (e.g. a trailing newline).
        let lines = raw
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)

        // The first line is the header row: it names the columns.
        guard let header = lines.first else { throw DatasetError.empty }
        let columns = parseCSVLine(header).map { $0.lowercased() }

        // We don't assume a fixed column order — we look up where 'label' and
        // 'text' actually are. `firstIndex(of:)` returns nil if not present.
        guard let labelIndex = columns.firstIndex(of: "label"),
              let textIndex = columns.firstIndex(of: "text") else {
            throw DatasetError.missingColumns
        }

        var points: [DataPoint] = []

        // `dropFirst()` skips the header; loop over the actual data rows.
        for line in lines.dropFirst() {
            let fields = parseCSVLine(line)

            // Defensive: make sure this row actually has the columns we need.
            guard fields.count > max(labelIndex, textIndex) else { continue }

            // The label is an integer (0 or 1). If it can't be parsed, skip the row.
            guard let label = Int(fields[labelIndex].trimmingCharacters(in: .whitespaces)) else { continue }

            let text = fields[textIndex]
            points.append(DataPoint(text: text, label: label))
        }

        guard !points.isEmpty else { throw DatasetError.empty }
        return points
    }

    /// Splits ONE CSV line into its fields.
    ///
    /// A naive `split(separator: ",")` is wrong for real text, because a field
    /// like `"Manchester wins 3, qualifies"` contains a comma *inside* quotes.
    /// This mini state-machine handles double-quoted fields and the CSV rule
    /// that a doubled quote `""` inside a quoted field means one literal quote.
    private func parseCSVLine(_ line: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var insideQuotes = false

        let chars = Array(line)   // index into the characters so we can peek ahead
        var i = 0
        while i < chars.count {
            let c = chars[i]

            if insideQuotes {
                if c == "\"" {
                    // A quote inside quotes: is it an escaped "" or the closing quote?
                    if i + 1 < chars.count && chars[i + 1] == "\"" {
                        current.append("\"")   // escaped double-quote -> one quote
                        i += 1                 // consume the second quote too
                    } else {
                        insideQuotes = false   // this quote closes the field
                    }
                } else {
                    current.append(c)
                }
            } else {
                switch c {
                case "\"":
                    insideQuotes = true        // opening quote
                case ",":
                    fields.append(current)     // comma ends the current field
                    current = ""
                default:
                    current.append(c)
                }
            }
            i += 1
        }

        fields.append(current)   // the last field has no trailing comma
        return fields
    }
}
