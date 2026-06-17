//
//  Classifier.swift
//  VisualML
//
//  Trains a linear classifier on the 2-D LSA points. All three kinds share one
//  gradient-descent loop; only the per-sample gradient differs.
//
//    score:        z = w·x + b
//    Logistic:     p = σ(z);  loss = cross-entropy;  grad = (p − y)·x
//    Linear (LSQ): ŷ = z;     loss = (z − y)²;       grad = (z − y)·x
//    SVM (hinge):  loss = max(0, 1 − ỹ·z), ỹ∈{−1,+1}; grad = −ỹ·x if ỹz < 1 else 0
//
//  We start from an ARBITRARY (random) line and record the boundary after every
//  recorded iteration, so the UI can replay the line converging onto the data.
//

import Foundation

struct Classifier {

    /// Train on the first two LSA coordinates of each document.
    func train(coords: [[Double]], labels: [Int],
               categories: [String], params: ClassifierParams) -> TrainedModel {
        let n = coords.count
        let X = coords.map { ($0[0], $0.count > 1 ? $0[1] : 0) }   // (component-1, component-2)
        let y = labels

        // ---- Deterministic shuffle + train/test split ----
        var idx = Array(0..<n)
        var splitRNG = SeededRNG(seed: 42)
        idx.shuffle(using: &splitRNG)
        let testN = max(1, Int(Double(n) * params.testFraction))
        let testIdx = Array(idx.prefix(testN))
        let testSet = Set(testIdx)
        let trainIdx = idx.filter { !testSet.contains($0) }

        // ---- Standardize features using TRAIN statistics ----
        func mean(_ sel: [Int], _ f: (Int) -> Double) -> Double {
            sel.reduce(0) { $0 + f($1) } / Double(sel.count)
        }
        let mu0 = mean(trainIdx) { X[$0].0 }, mu1 = mean(trainIdx) { X[$0].1 }
        let sd0 = sqrt(max(mean(trainIdx) { pow(X[$0].0 - mu0, 2) }, 1e-12))
        let sd1 = sqrt(max(mean(trainIdx) { pow(X[$0].1 - mu1, 2) }, 1e-12))
        func feat(_ i: Int) -> (Double, Double) { ((X[i].0 - mu0) / sd0, (X[i].1 - mu1) / sd1) }

        let threshold: Double = (params.kind == .linear) ? 0.5 : 0.0   // linear regresses toward 0/1

        // Predict / accuracy / coordinate-conversion for ANY weights (used per frame).
        func predict(_ i: Int, _ w0: Double, _ w1: Double, _ b: Double) -> Int {
            let (x0, x1) = feat(i)
            return (w0 * x0 + w1 * x1 + b) >= threshold ? 1 : 0
        }
        func accuracy(_ sel: [Int], _ w0: Double, _ w1: Double, _ b: Double) -> Double {
            guard !sel.isEmpty else { return 0 }
            return Double(sel.reduce(0) { $0 + (predict($1, w0, w1, b) == y[$1] ? 1 : 0) }) / Double(sel.count)
        }
        // Standardized boundary z = threshold → a line w0o·x + w1o·y + bo = 0 in LSA coords.
        func toLSA(_ w0: Double, _ w1: Double, _ b: Double) -> (Double, Double, Double) {
            let a = w0 / sd0, c = w1 / sd1
            return (a, c, (b - threshold) - (a * mu0 + c * mu1))
        }

        // ---- Start from an ARBITRARY line ----
        var initRNG = SeededRNG(seed: 99)
        var w0 = Double(initRNG.next() % 1000) / 1000.0 - 0.5
        var w1 = Double(initRNG.next() % 1000) / 1000.0 - 0.5
        var b  = Double(initRNG.next() % 1000) / 1000.0 - 0.5

        let eta = params.learningRate
        let lambda = params.regularization
        let iters = Int(params.iterations)
        let kind = params.kind
        let m = Double(max(trainIdx.count, 1))

        // Record at most ~120 frames so playback is smooth and bounded.
        let frameEvery = max(1, iters / 120)
        var history: [BoundaryStep] = []
        func record(_ it: Int) {
            let (a, c, d) = toLSA(w0, w1, b)
            history.append(BoundaryStep(iteration: it, w0: a, w1: c, b: d,
                                        trainAccuracy: accuracy(trainIdx, w0, w1, b)))
        }
        record(0)   // the arbitrary starting line

        for it in 0..<iters {
            var g0 = 0.0, g1 = 0.0, gb = 0.0
            for i in trainIdx {
                let (x0, x1) = feat(i)
                let z = w0 * x0 + w1 * x1 + b
                let yi = Double(y[i])
                switch kind {
                case .logistic:
                    let e = (1 / (1 + exp(-z))) - yi
                    g0 += e * x0; g1 += e * x1; gb += e
                case .linear:
                    let e = z - yi
                    g0 += e * x0; g1 += e * x1; gb += e
                case .svm:
                    let yt = 2 * yi - 1
                    if yt * z < 1 { g0 += -yt * x0; g1 += -yt * x1; gb += -yt }
                }
            }
            g0 /= m; g1 /= m; gb /= m
            g0 += lambda * w0; g1 += lambda * w1
            w0 -= eta * g0; w1 -= eta * g1; b -= eta * gb

            if (it + 1) % frameEvery == 0 || it == iters - 1 { record(it + 1) }
        }

        // ---- Final metrics ----
        var confusion = [[0, 0], [0, 0]]
        for i in testIdx { confusion[y[i]][predict(i, w0, w1, b)] += 1 }
        let (w0o, w1o, bo) = toLSA(w0, w1, b)
        var c0 = "0", c1 = "1"
        for i in 0..<n { if y[i] == 0 { c0 = categories[i] } else if y[i] == 1 { c1 = categories[i] } }

        return TrainedModel(kind: kind, w0: w0o, w1: w1o, b: bo,
                            drawMargins: kind == .svm,
                            trainAccuracy: accuracy(trainIdx, w0, w1, b),
                            testAccuracy: accuracy(testIdx, w0, w1, b),
                            confusion: confusion,
                            class0: c0, class1: c1, testCount: testIdx.count,
                            history: history)
    }
}

/// A tiny seedable RNG (SplitMix64) so training is reproducible.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    nonisolated init(seed: UInt64) { state = seed }
    nonisolated mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
