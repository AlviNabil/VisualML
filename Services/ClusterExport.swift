//
//  ClusterExport.swift
//  VisualML
//
//  Decodes the k-means results precomputed by ml/make_kmeans.py. The app fits
//  nothing itself: it reads the customers, the elbow curve, the recorded
//  restart histories and the final clusters out of the bundled JSON.
//

import Foundation

/// One customer: the feature values and the segment the data was generated
/// from (unused by k-means, kept only to check the clusters against reality).
struct ClusterPoint: Codable {
    let x: [Double]
    let trueSegment: String
}

/// The dataset a clustering was fit on.
struct ClusterDataset: Codable {
    let name: String
    let featureNames: [String]
    let axisLabels: [String]
    let n: Int
    let mins: [Double]
    let maxs: [Double]
    let points: [ClusterPoint]
}

/// The best-of-restarts result at one value of k, used to draw the elbow curve
/// and to preview what the clusters look like at that k.
struct ElbowPoint: Codable, Identifiable {
    let k: Int
    let inertia: Double
    let centroids: [[Double]]
    let assignments: [Int]

    var id: Int { k }
}

/// One recorded step of k-means: the assign-then-average loop.
struct ClusterFrame: Codable {
    let iter: Int
    let centroids: [[Double]]
    let assignments: [Int]
    let inertia: Double
}

/// One full k-means run from a k-means++ start, at a fixed random seed.
struct ClusterRestart: Codable, Identifiable {
    let seed: Int
    let iterations: Int
    let finalInertia: Double
    let purity: Double
    let history: [ClusterFrame]

    var id: Int { seed }
}

/// The true segment a cluster's members mostly belong to, for labelling the
/// result -- computed after fitting, never used by the algorithm itself.
struct ClusterLabel: Codable, Identifiable {
    let cluster: Int
    let majoritySegment: String?
    let matchRate: Double
    let size: Int

    var id: Int { cluster }
}

/// The lowest-inertia restart: the final answer the app leads with.
struct ClusterBest: Codable {
    let seed: Int
    let centroids: [[Double]]
    let assignments: [Int]
    let inertia: Double
    let purity: Double
    let clusterLabels: [ClusterLabel]
}

/// The whole precomputed export for the k-means stage.
struct ClusterExport: Codable {
    let model: String
    let title: String
    let subtitle: String
    let library: String
    let k: Int
    let dataset: ClusterDataset
    let elbow: [ElbowPoint]
    let restarts: [ClusterRestart]
    let best: ClusterBest

    /// Loads and decodes `kmeans.json` from the app bundle.
    static func load() throws -> ClusterExport {
        guard let url = Bundle.main.url(forResource: "kmeans", withExtension: "json") else {
            throw RegressionExportError.notFound("kmeans")
        }
        return try JSONDecoder().decode(ClusterExport.self, from: try Data(contentsOf: url))
    }
}
