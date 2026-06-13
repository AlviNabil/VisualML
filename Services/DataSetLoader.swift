//
//  DatasetLoader.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import Foundation
import CreateML

class DatasetLoader {
    // Loads the CSV and checks for the correct format
    func loadCSV(filename: String) async throws -> [DataPoint] {
        guard let path = Bundle.main.path(forResource: filename, ofType: "csv") else {
            throw URLError(.fileDoesNotExist)
        }
        
        let dataFrame = try MLDataTable(contentsOf: URL(fileURLWithPath: path))
        
        guard dataFrame.columnNames.contains("label") && dataFrame.columnNames.contains("text") else {
            throw URLError(.cannotParseResponse)
        }
        
        var points: [DataPoint] = []
        
        // 1. Extract the columns first
        let textColumn = dataFrame["text"]
        let labelColumn = dataFrame["label"]
        
        // 2. Specify .rows from the size tuple
        for row in 0..<dataFrame.size.rows {
            // 3. Access the row from the specific column
            if let text = textColumn[row].stringValue,
               let label = labelColumn[row].intValue {
                points.append(DataPoint(text: text, label: label))
            }
        }
        
        return points
    }
}
