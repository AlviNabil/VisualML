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
        
        Task {
            do {
                // Change "dataset" to whatever you name your CSV file
                self.dataPoints = try await loader.loadCSV(filename: "dataset")
                self.statusMessage = "Successfully loaded \(self.dataPoints.count) items."
            } catch {
                self.statusMessage = "Error: Please check your CSV format."
            }
        }
    }
}
