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
            return "The CSV must contain both a 'category' and a 'text' column."
        case .empty:
            return "The dataset has no usable rows."
        }
    }
}

/// Loads a bundled CSV of labeled text into an array of `DataPoint`.
///
/// We parse the CSV by hand instead of using Apple's `CreateML` / `MLDataTable`,
/// for two reasons:
///   1. CreateML is a *macOS-only* framework — importing it breaks the iOS build.
///   2. Doing it ourselves keeps the whole pipeline transparent, which is the
///      entire point of this teaching app.
///
/// Expected CSV shape:
///     category,text
///     sport,"The striker scored a hat-trick ..."
///     business,"The company reported record earnings ..."
class DatasetLoader {

    /// Reads `<filename>.csv` from the app bundle and returns the parsed rows.
    /// - Parameter filename: the resource name WITHOUT the `.csv` extension.
    func loadCSV(filename: String) throws -> [DataPoint] {
        // `Bundle.main` is the running app's container. `url(forResource:...)`
        // finds a file we added to the "Copy Bundle Resources" build phase.
        guard let url = Bundle.main.url(forResource: filename, withExtension: "csv") else {
            throw DatasetError.fileNotFound(filename)
        }

        // Read the whole file into one big String, then split into lines.
        let raw = try String(contentsOf: url, encoding: .utf8)
        let lines = raw
            .replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)

        // The first line is the header: it names the columns.
        guard let header = lines.first else { throw DatasetError.empty }
        let columns = parseCSVLine(header).map { $0.lowercased() }

        // We don't assume column order — we look up where each column actually is.
        guard let categoryIndex = columns.firstIndex(of: "category"),
              let textIndex = columns.firstIndex(of: "text") else {
            throw DatasetError.missingColumns
        }

        // PASS 1 — pull out the raw (category, text) pairs.
        var rawRows: [(category: String, text: String)] = []
        for line in lines.dropFirst() {                 // dropFirst() skips the header
            let fields = parseCSVLine(line)
            guard fields.count > max(categoryIndex, textIndex) else { continue }

            let category = fields[categoryIndex].trimmingCharacters(in: .whitespaces)
            let text = fields[textIndex].trimmingCharacters(in: .whitespaces)
            guard !category.isEmpty, !text.isEmpty else { continue }

            rawRows.append((category, text))
        }
        guard !rawRows.isEmpty else { throw DatasetError.empty }

        // Build a STABLE label map: take the distinct category names, sort them
        // alphabetically, and number them 0, 1, 2 ... So {business, sport} always
        // becomes business = 0, sport = 1, no matter what order the rows are in.
        let classNames = Set(rawRows.map { $0.category }).sorted()
        let labelOf = Dictionary(uniqueKeysWithValues:
            classNames.enumerated().map { (index, name) in (name, index) })

        // PASS 2 — build DataPoints carrying both the readable name and the index.
        return rawRows.map { row in
            DataPoint(text: row.text, category: row.category, label: labelOf[row.category]!)
        }
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
                    // A quote inside quotes: escaped "" or the closing quote?
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
