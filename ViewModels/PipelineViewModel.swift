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
    
    private let loader = DatasetLoader()
    
    func loadDataset() {
        statusMessage = "Loading..."

        do {
            // "dataset" matches dataset.csv in the app bundle.
            dataPoints = try loader.loadCSV(filename: "dataset")
            statusMessage = "Loaded \(dataPoints.count) documents."
        } catch {
            // `localizedDescription` comes from our DatasetError.errorDescription,
            // so this tells us exactly what went wrong.
            statusMessage = "Error: \(error.localizedDescription)"
        }
    }
}
