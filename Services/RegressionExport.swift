//
//  RegressionExport.swift
//  VisualML
//
//  Decodes the regression results precomputed by the Python scripts in ml/.
//  The app performs no fitting of its own: it reads the parameters, the metrics
//  and the recorded gradient-descent history out of the bundled JSON and draws
//  them.
//

import Foundation

/// One row of the dataset: the feature values and the observed target.
struct RegressionPoint: Codable {
    let x: [Double]        // one entry per feature
    let y: Double
    let test: Bool         // true when the row was held out of training
}

/// The dataset a model was fit on, plus the ranges needed to scale the plots.
struct RegressionDataset: Codable {
    let name: String
    let featureNames: [String]
    let axisLabels: [String]
    let targetLabel: String
    let n: Int
    let mins: [Double]
    let maxs: [Double]
    let targetMin: Double
    let targetMax: Double
    let trainCount: Int
    let testCount: Int
    let points: [RegressionPoint]
}

/// A fitted parameter vector with its scores. `theta` is [w1 ... wd, b].
struct RegressionSolution: Codable {
    let theta: [Double]
    let mse: Double
    let rmse: Double
    let r2: Double
    let testMse: Double
    let testR2: Double

    /// The feature weights, excluding the trailing intercept.
    var weights: [Double] { Array(theta.dropLast()) }
    /// The intercept term.
    var intercept: Double { theta.last ?? 0 }

    /// Evaluate the model for one row of feature values.
    func predict(_ x: [Double]) -> Double {
        zip(weights, x).reduce(intercept) { $0 + $1.0 * $1.1 }
    }
}

/// The score for always predicting the training mean.
struct RegressionBaseline: Codable {
    let prediction: Double
    let mse: Double
}

/// One recorded step of gradient descent.
struct RegressionFrame: Codable {
    let iter: Int
    let theta: [Double]
    let mse: Double
    let r2: Double
    let testMse: Double
    let testR2: Double

    var weights: [Double] { Array(theta.dropLast()) }
    var intercept: Double { theta.last ?? 0 }

    func predict(_ x: [Double]) -> Double {
        zip(weights, x).reduce(intercept) { $0 + $1.0 * $1.1 }
    }
}

/// One complete gradient-descent run at a fixed learning rate.
struct RegressionRun: Codable, Identifiable {
    let learningRate: Double
    let fractionOfLimit: Double
    let iterations: Int
    let diverged: Bool
    let finalSolution: RegressionSolution?
    let history: [RegressionFrame]

    var id: Double { learningRate }

    enum CodingKeys: String, CodingKey {
        case learningRate, fractionOfLimit, iterations, diverged, history
        case finalSolution = "final"
    }

    /// Short label describing this rate relative to the stability limit.
    var label: String {
        String(format: "%.2f×", fractionOfLimit)
    }
}

/// The whole precomputed export for one regression model.
struct RegressionExport: Codable {
    let model: String
    let featureCount: Int
    let title: String
    let subtitle: String
    let library: String
    let dataset: RegressionDataset
    let closedForm: RegressionSolution
    let baseline: RegressionBaseline
    let residuals: [Double]
    let runs: [RegressionRun]

    /// Loads and decodes `<name>.json` from the app bundle.
    static func load(_ name: String) throws -> RegressionExport {
        guard let url = Bundle.main.url(forResource: name, withExtension: "json") else {
            throw RegressionExportError.notFound(name)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(RegressionExport.self, from: data)
    }
}

/// Formats a score for a metric card, switching to scientific notation for the
/// extreme magnitudes a diverging run produces.
func formatScore(_ value: Double) -> String {
    guard value.isFinite else { return "—" }
    return abs(value) >= 1000
        ? String(format: "%.1e", value)
        : String(format: "%.3f", value)
}

enum RegressionExportError: Error, LocalizedError {
    case notFound(String)

    var errorDescription: String? {
        switch self {
        case .notFound(let name):
            return "Could not find \(name).json in the app bundle."
        }
    }
}
