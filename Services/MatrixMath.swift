//
//  MatrixMath.swift
//  VisualML
//
//  The linear algebra for LSA. We factorize the (weighted) document-term matrix
//  with SVD, keeping the top components — this is Latent Semantic Analysis.
//
//  The symmetric eigendecomposition is a hand-written Jacobi solver (no LAPACK /
//  Accelerate dependency, fully transparent). Performance notes:
//    - Flat [Double] storage (i*n + j) avoids array-of-arrays overhead.
//    - A RELATIVE convergence test stops after ~10 sweeps instead of always
//      running the 100-sweep cap.
//

import Foundation

struct MatrixMath {

    /// Latent Semantic Analysis via the Gram matrix G = W·Wᵀ.
    ///
    /// W = U·Σ·Vᵀ ⇒ W·Wᵀ = U·Σ²·Uᵀ, so eigenvectors of the small D×D Gram matrix
    /// are U and eigenvalues are σ². Document coordinates = rows of U·Σ; term
    /// loadings (which words define a component) are V = Wᵀ·U·Σ⁻¹.
    nonisolated func performLSA(_ w: WeightedMatrix, components k: Int = 2) -> SVDResult {
        let docCount = w.rows.count          // (avoid the @MainActor computed props)
        let vocabCount = w.vocab.count
        guard docCount > 0, vocabCount > 0 else {
            return SVDResult(coords: [], singularValues: [], spectrum: [],
                             termLoadings: [], vocab: w.vocab,
                             labels: w.labels, categories: w.categories)
        }

        // 1. Gram matrix G = W·Wᵀ (D×D), stored flat.
        var g = [Double](repeating: 0, count: docCount * docCount)
        for i in 0..<docCount {
            let ri = w.rows[i]
            for j in i..<docCount {
                let rj = w.rows[j]
                var dot = 0.0
                for t in 0..<vocabCount { dot += ri[t] * rj[t] }
                g[i * docCount + j] = dot
                g[j * docCount + i] = dot
            }
        }

        // 2. Symmetric eigendecomposition (Jacobi).
        let (eigenvalues, eigenvectors) = jacobiEigen(&g, docCount)

        // 3. Order components by eigenvalue (largest first) + singular values.
        let order = (0..<docCount).sorted { eigenvalues[$0] > eigenvalues[$1] }
        let spectrum = order.map { sqrt(max(eigenvalues[$0], 0)) }

        // 4. Document coordinates: coord[d][c] = U[d, col]·σ_c.
        let kept = min(k, docCount)
        var coords = [[Double]](repeating: [Double](repeating: 0, count: kept), count: docCount)
        var topSV = [Double](repeating: 0, count: kept)
        for c in 0..<kept {
            let col = order[c]
            let sigma = sqrt(max(eigenvalues[col], 0))
            topSV[c] = sigma
            for d in 0..<docCount {
                coords[d][c] = eigenvectors[d * docCount + col] * sigma
            }
        }

        // 5. Term loadings: loading[t][c] = (Σ_d W[d][t]·U[d][col]) / σ_c.
        var loadings = [[Double]](repeating: [Double](repeating: 0, count: kept), count: vocabCount)
        for c in 0..<kept where topSV[c] > 0 {
            let col = order[c]
            let sigma = topSV[c]
            for t in 0..<vocabCount {
                var s = 0.0
                for d in 0..<docCount { s += w.rows[d][t] * eigenvectors[d * docCount + col] }
                loadings[t][c] = s / sigma
            }
        }

        return SVDResult(coords: coords, singularValues: topSV, spectrum: spectrum,
                         termLoadings: loadings, vocab: w.vocab,
                         labels: w.labels, categories: w.categories)
    }

    /// Cyclic Jacobi eigenvalue algorithm for a symmetric matrix stored flat
    /// (row-major, n×n). Returns (eigenvalues, eigenvectors) where eigenvector
    /// `c` is column c: entry i is `vectors[i*n + c]`.
    nonisolated private func jacobiEigen(_ a: inout [Double], _ n: Int) -> (values: [Double], vectors: [Double]) {
        var v = [Double](repeating: 0, count: n * n)
        for i in 0..<n { v[i * n + i] = 1 }

        // Frobenius² is invariant under the rotations; use it to scale a RELATIVE
        // stop test, so we don't spin for the full sweep cap.
        var frob = 0.0
        for x in a { frob += x * x }
        let threshold = 1e-14 * max(frob, 1e-300)

        for _ in 0..<100 {
            var off = 0.0
            for p in 0..<n {
                let rp = p * n
                for q in (p + 1)..<n { let x = a[rp + q]; off += x * x }
            }
            if off <= threshold { break }

            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    let apq = a[p * n + q]
                    if apq == 0 { continue }
                    let app = a[p * n + p], aqq = a[q * n + q]
                    let tau = (aqq - app) / (2 * apq)
                    let t = (tau >= 0 ? 1.0 : -1.0) / (abs(tau) + (tau * tau + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c

                    // A ← Jᵀ A J : rotate columns p,q then rows p,q.
                    for i in 0..<n {
                        let ip = i * n + p, iq = i * n + q
                        let aip = a[ip], aiq = a[iq]
                        a[ip] = c * aip - s * aiq
                        a[iq] = s * aip + c * aiq
                    }
                    let rp = p * n, rq = q * n
                    for i in 0..<n {
                        let api = a[rp + i], aqi = a[rq + i]
                        a[rp + i] = c * api - s * aqi
                        a[rq + i] = s * api + c * aqi
                    }
                    // V ← V J : accumulate the rotation into the eigenvectors.
                    for i in 0..<n {
                        let ip = i * n + p, iq = i * n + q
                        let vip = v[ip], viq = v[iq]
                        v[ip] = c * vip - s * viq
                        v[iq] = s * vip + c * viq
                    }
                }
            }
        }

        var values = [Double](repeating: 0, count: n)
        for i in 0..<n { values[i] = a[i * n + i] }
        return (values, v)
    }
}
