//
//  NeuralExport.swift
//  VisualML
//
//  Decodes the neural network results precomputed by ml/make_neural.py. The
//  app trains nothing: it reads the weights, the per-layer forward values, the
//  gradients, the activation-function samples and the recorded training run
//  out of the bundled JSON.
//

import Foundation

/// One row of the dataset: two features and the class it belongs to.
struct NeuralPoint: Codable {
    let x: [Double]
    let label: Int
}

/// The dataset, plus how badly a single straight boundary does on it.
struct NeuralDataset: Codable {
    let name: String
    let featureNames: [String]
    let axisLabels: [String]
    let classNames: [String]
    let n: Int
    let min: Double
    let max: Double
    let points: [NeuralPoint]
    let linearBaselineAccuracy: Double
}

/// The shape of the network and how it was trained.
struct NeuralArchitecture: Codable {
    let layerSizes: [Int]
    let hiddenActivation: String
    let outputActivation: String
    let learningRate: Double
    let epochs: Int
    let parameterCount: Int

    var inputCount: Int { layerSizes.first ?? 0 }
    var outputCount: Int { layerSizes.last ?? 0 }
    var hiddenSizes: [Int] { Array(layerSizes.dropFirst().dropLast()) }
    /// Weight-matrix shapes, one per layer, as (rows, columns).
    var weightShapes: [(Int, Int)] {
        zip(layerSizes.dropLast(), layerSizes.dropFirst()).map { ($0, $1) }
    }
}

/// One layer of a single point's journey forward through the network.
struct ForwardStep: Codable, Identifiable {
    let layer: Int
    let activation: String
    let input: [Double]
    let weights: [[Double]]
    let biases: [Double]
    let z: [Double]
    let a: [Double]

    var id: Int { layer }
}

/// One point carried all the way through, with every intermediate kept.
struct ForwardTrace: Codable {
    let input: [Double]
    let label: Int
    let steps: [ForwardStep]
    let prediction: Double
    let loss: Double
}

/// One layer of the error signal's journey back through the network.
struct BackwardStep: Codable, Identifiable {
    let layer: Int
    let dz: [Double]
    let dW: [[Double]]
    let db: [Double]
    /// Present for every layer except the first: the error arriving from the
    /// layer above, before it is scaled by this layer's activation slope.
    let daPrev: [Double]?
    /// f'(z) for the layer below — the term that decides how much of the
    /// signal survives the trip backward.
    let primeZ: [Double]?

    var id: Int { layer }
}

/// The gradients for one point, ordered from the output layer backward.
struct BackwardTrace: Codable {
    let prediction: Double
    let label: Int
    let outputError: Double
    let steps: [BackwardStep]
    let epoch: Int
}

/// How one activation function performed when the whole net was trained on it.
struct ActivationTraining: Codable {
    let finalLoss: Double
    let finalAccuracy: Double
    /// Nil if the run never reached 95% accuracy.
    let epochsTo95: Int?
    let curve: [LossPoint]
}

/// One activation function: its shape, its slope, and how fast it learns.
struct ActivationSample: Codable, Identifiable {
    let name: String
    let formula: String
    let derivative: String
    let range: String
    let note: String
    let x: [Double]
    let y: [Double]
    let dy: [Double]
    let maxSlope: Double
    let training: ActivationTraining

    var id: String { name }

    /// "leaky_relu" reads better as "Leaky ReLU" on screen.
    var displayName: String {
        switch name {
        case "relu": return "ReLU"
        case "leaky_relu": return "Leaky ReLU"
        case "elu": return "ELU"
        case "gelu": return "GELU"
        default: return name.capitalized
        }
    }
}

/// One point on a loss curve.
struct LossPoint: Codable {
    let epoch: Int
    let loss: Double
    let accuracy: Double?
}

/// One recorded checkpoint of training: the weights, the scores, and the
/// probability the network assigns across a grid covering the whole plane.
struct TrainingFrame: Codable, Identifiable {
    let epoch: Int
    let loss: Double
    let accuracy: Double
    let weights: [[[Double]]]
    let biases: [[Double]]
    let grid: [[Double]]

    var id: Int { epoch }
}

/// The whole recorded training run.
struct NeuralTraining: Codable {
    let gridSize: Int
    let gridMin: Double
    let gridMax: Double
    let curve: [LossPoint]
    let frames: [TrainingFrame]
    let finalLoss: Double
    let finalAccuracy: Double
}

/// What one hidden unit responds to, sampled across the plane.
struct NeuronResponse: Codable, Identifiable {
    let layer: Int
    let unit: Int
    let grid: [[Double]]

    var id: String { "\(layer)-\(unit)" }
}

/// The whole precomputed export for the neural network stage.
struct NeuralExport: Codable {
    let model: String
    let title: String
    let subtitle: String
    let library: String
    let dataset: NeuralDataset
    let architecture: NeuralArchitecture
    let forwardTrace: ForwardTrace
    let forwardTraceAtBackprop: ForwardTrace
    let backwardTrace: BackwardTrace
    let activations: [ActivationSample]
    let training: NeuralTraining
    let neurons: [NeuronResponse]

    /// Loads and decodes `neural.json` from the app bundle.
    static func load() throws -> NeuralExport {
        guard let url = Bundle.main.url(forResource: "neural", withExtension: "json") else {
            throw RegressionExportError.notFound("neural")
        }
        return try JSONDecoder().decode(NeuralExport.self, from: try Data(contentsOf: url))
    }
}
