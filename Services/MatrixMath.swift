//
//  MatrixMath.swift
//  VisualML
//
//  The linear algebra for LSA. We need only the top few components, so instead
//  of a full eigendecomposition we use POWER ITERATION with deflation: repeatedly
//  apply the Gram operator G·v = W·(Wᵀ·v) and renormalize; the vector converges
//  to the largest eigenvector. After finding one we deflate it and repeat.
//
//  The hot loops use a single flat row-major buffer + unsafe pointers (no D×D
//  matrix is ever built, no bounds checks), so it stays fast at ~500 documents.
//

import Foundation

struct MatrixMath {

    /// LSA of the weighted matrix W (D documents × V terms).
    /// Eigenvectors of G = W·Wᵀ are the left singular vectors U; eigenvalues are σ².
    /// Document coordinates = rows of U·Σ; term loadings = (Wᵀ·U)/σ.
    nonisolated func performLSA(_ w: WeightedMatrix, components k: Int = 2) -> SVDResult {
        let docCount = w.rows.count
        let vocabCount = w.vocab.count
        guard docCount > 0, vocabCount > 0 else {
            return SVDResult(coords: [], singularValues: [], spectrum: [],
                             termLoadings: [], vocab: w.vocab,
                             labels: w.labels, categories: w.categories)
        }
        let topN = min(max(k, 4), docCount)   // 2 for the plot + a couple for the scree
        let kept = min(k, topN)
        let flat = w.rows.flatMap { $0 }      // D×V, row-major contiguous

        var spectrum = [Double]()
        var topSV = [Double](repeating: 0, count: kept)
        var coords = [[Double]](repeating: [Double](repeating: 0, count: kept), count: docCount)
        var loadings = [[Double]](repeating: [Double](repeating: 0, count: kept), count: vocabCount)

        flat.withUnsafeBufferPointer { F in
            var foundVals = [Double]()
            var foundVecs = [[Double]]()

            // u = Wᵀ·v  (length V).  W is `flat` (row-major), so column t of row d is F[d*V + t].
            func wTransposeTimes(_ v: [Double]) -> [Double] {
                var u = [Double](repeating: 0, count: vocabCount)
                v.withUnsafeBufferPointer { vp in
                    u.withUnsafeMutableBufferPointer { up in
                        for d in 0..<docCount {
                            let vd = vp[d]
                            if vd != 0 {
                                let base = d * vocabCount
                                for t in 0..<vocabCount { up[t] += F[base + t] * vd }
                            }
                        }
                    }
                }
                return u
            }

            // G·v = W·(Wᵀ·v), minus the deflated (already-found) directions.
            func gramApply(_ v: [Double]) -> [Double] {
                let u = wTransposeTimes(v)
                var r = [Double](repeating: 0, count: docCount)
                u.withUnsafeBufferPointer { up in
                    r.withUnsafeMutableBufferPointer { rp in
                        for d in 0..<docCount {
                            let base = d * vocabCount
                            var s = 0.0
                            for t in 0..<vocabCount { s += F[base + t] * up[t] }
                            rp[d] = s
                        }
                    }
                }
                for c in 0..<foundVals.count {
                    let vec = foundVecs[c]
                    var proj = 0.0
                    for i in 0..<docCount { proj += vec[i] * v[i] }
                    let scale = foundVals[c] * proj
                    for i in 0..<docCount { r[i] -= scale * vec[i] }
                }
                return r
            }

            var rng = SeededRNG(seed: 0xA17EC)
            for _ in 0..<topN {
                var v = (0..<docCount).map { _ in Double(rng.next() % 2000) / 1000.0 - 1.0 }
                normalize(&v)
                var lambda = 0.0
                for _ in 0..<150 {                          // top-2 converge in ~40
                    let r = gramApply(v)
                    var rayleigh = 0.0, normSq = 0.0
                    for i in 0..<docCount { rayleigh += v[i] * r[i]; normSq += r[i] * r[i] }
                    let norm = normSq.squareRoot()
                    if norm < 1e-12 { lambda = max(rayleigh, 0); break }
                    for i in 0..<docCount { v[i] = r[i] / norm }
                    if abs(rayleigh - lambda) <= 1e-9 * abs(rayleigh) + 1e-12 { lambda = rayleigh; break }
                    lambda = rayleigh
                }
                foundVals.append(max(lambda, 0))
                foundVecs.append(v)
            }

            spectrum = foundVals.map { $0.squareRoot() }
            for c in 0..<kept {
                let sigma = foundVals[c].squareRoot()
                topSV[c] = sigma
                let vec = foundVecs[c]
                for d in 0..<docCount { coords[d][c] = vec[d] * sigma }   // U·Σ
                if sigma > 0 {
                    let u = wTransposeTimes(vec)                          // Wᵀ·U_c
                    for t in 0..<vocabCount { loadings[t][c] = u[t] / sigma }
                }
            }
        }

        return SVDResult(coords: coords, singularValues: topSV, spectrum: spectrum,
                         termLoadings: loadings, vocab: w.vocab,
                         labels: w.labels, categories: w.categories)
    }

    private nonisolated func normalize(_ v: inout [Double]) {
        var norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        if norm == 0 { v[0] = 1; norm = 1 }
        for i in v.indices { v[i] /= norm }
    }
}
