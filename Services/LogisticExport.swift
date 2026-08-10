//
//  LogisticExport.swift
//  VisualML
//
//  Decodes the logistic regression results precomputed by ml/make_logistic.py.
//  The app fits nothing: it reads the parameters, the sampled curve, the loss
//  landscape and the recorded training history out of the bundled JSON.
//

import Foundation

/// One student: the feature values and whether they passed.
struct LogisticPoint: Codable {
    let x: [Double]
    let y: Int              // 0 = fail, 1 = pass
    let test: Bool
}

/// The dataset, plus the ranges and class names the plots need.
struct LogisticDataset: Codable {
    let name: String
    let featureNames: [String]
    let axisLabels: [String]
    let targetLabel: String
    let classNames: [String]     // [negative, positive]
    let n: Int
    let mins: [Double]
    let maxs: [Double]
    let trainCount: Int
    let testCount: Int
    let positiveCount: Int
    let negativeCount: Int
    let points: [LogisticPoint]
}

/// A least-squares line fit to the 0/1 labels, kept for contrast.
struct LinearBaseline: Codable {
    let theta: [Double]
    let outOfRangeCount: Int
    let minPrediction: Double
    let maxPrediction: Double

    var slope: Double { theta.first ?? 0 }
    var intercept: Double { theta.last ?? 0 }

    func predict(_ x: Double) -> Double { slope * x + intercept }
}

/// The converged model and how well it scores.
struct LogisticFit: Codable {
    let learningRate: Double
    let theta: [Double]
    let logLoss: Double
    let accuracy: Double
    let testAccuracy: Double
    let boundary: Double         // the x where p = 0.5
    let oddsRatio: Double        // e^w — the odds multiplier per unit of x
    let confusion: [[Int]]
    let testConfusion: [[Int]]
    let majorityAccuracy: Double

    var weight: Double { theta.first ?? 0 }
    var intercept: Double { theta.last ?? 0 }
}

/// One sample of the fitted model: the raw score and the squashed probability.
struct CurveSample: Codable {
    let x: Double
    let z: Double
    let p: Double
}

/// The share of students who actually passed within one band of the feature.
struct EmpiricalRate: Codable, Identifiable {
    let binStart: Double
    let binEnd: Double
    let center: Double
    let count: Int
    let passed: Int
    let rate: Double

    var id: Double { center }
}

/// How the model scores when the cut between the two classes is moved.
struct ThresholdPoint: Codable, Identifiable {
    let threshold: Double
    let accuracy: Double
    let testAccuracy: Double
    let confusion: [[Int]]
    let testConfusion: [[Int]]
    let precision: Double
    let recall: Double

    var id: Double { threshold }
}

/// Cross-entropy sampled over a grid of (w, b). `values[bIndex][wIndex]`.
struct LossSurface: Codable {
    let wMin: Double, wMax: Double
    let bMin: Double, bMax: Double
    let steps: Int
    let values: [[Double]]

    var lowest: Double { values.flatMap { $0 }.min() ?? 0 }
    var highest: Double { values.flatMap { $0 }.max() ?? 1 }
}

/// One recorded step of gradient descent.
struct LogisticFrame: Codable {
    let iter: Int
    let theta: [Double]
    let grad: [Double]
    let logLoss: Double
    let accuracy: Double
    let testAccuracy: Double
    let boundary: Double?

    var weight: Double { theta.first ?? 0 }
    var intercept: Double { theta.last ?? 0 }

    /// The raw score for a feature value.
    func score(_ x: Double) -> Double { weight * x + intercept }
    /// The predicted probability for a feature value.
    func probability(_ x: Double) -> Double { sigmoid(score(x)) }
}

/// One training run at a fixed learning rate.
struct LogisticRun: Codable, Identifiable {
    let learningRate: Double
    let iterations: Int
    let diverged: Bool
    let finalFrame: LogisticSummary?
    let history: [LogisticFrame]

    var id: Double { learningRate }

    enum CodingKeys: String, CodingKey {
        case learningRate, iterations, diverged, history
        case finalFrame = "final"
    }
}

/// The end state of a run.
struct LogisticSummary: Codable {
    let theta: [Double]
    let logLoss: Double
    let accuracy: Double
    let testAccuracy: Double
    let boundary: Double?
}

/// The whole precomputed export for the logistic stage.
struct LogisticExport: Codable {
    let model: String
    let title: String
    let subtitle: String
    let library: String
    let dataset: LogisticDataset
    let linearBaseline: LinearBaseline
    let fit: LogisticFit
    let curve: [CurveSample]
    let empiricalRate: [EmpiricalRate]
    let thresholdSweep: [ThresholdPoint]
    let lossSurface: LossSurface
    let runs: [LogisticRun]

    /// Loads and decodes `logistic_regression.json` from the app bundle.
    static func load() throws -> LogisticExport {
        guard let url = Bundle.main.url(forResource: "logistic_regression",
                                        withExtension: "json") else {
            throw RegressionExportError.notFound("logistic_regression")
        }
        return try JSONDecoder().decode(LogisticExport.self,
                                        from: try Data(contentsOf: url))
    }
}

/// σ(z) = 1 / (1 + e^−z), the squash that turns a score into a probability.
func sigmoid(_ z: Double) -> Double {
    z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 + exp(z))
}
