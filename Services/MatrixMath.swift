//
//  MatrixMath.swift
//  VisualML
//
//  The linear algebra for LSA. We factorize the (weighted) document-term matrix
//  with SVD, keeping the top components — this is Latent Semantic Analysis.
//
//  We compute the symmetric eigendecomposition ourselves with the classic
//  Jacobi rotation method instead of calling LAPACK. Reasons:
//    1. Zero dependencies / no deprecated CLAPACK warnings.
//    2. It's fully transparent — fitting for a glass-box teaching app.
//  For our small Gram matrix (D ≈ 100) it runs in a few milliseconds.
//

import Foundation

struct MatrixMath {

    /// Latent Semantic Analysis via the **Gram matrix**.
    ///
    /// The math (see the master plan §4.4): for W (D documents × V terms),
    ///   W = U · Σ · Vᵀ              (SVD)
    ///   W · Wᵀ = U · Σ² · Uᵀ        (so the eigenvectors of the D×D Gram matrix
    ///                                are U, and the eigenvalues are σ²)
    /// Because D (≈100) is far smaller than V, eigendecomposing the small D×D
    /// Gram matrix is much cheaper than a full V×V SVD. The document coordinates
    /// we plot are the rows of U·Σ.
    ///
    /// - Parameters:
    ///   - w: the weighted matrix (raw counts or TF-IDF).
    ///   - k: how many leading components to keep (2 for the scatter plot).
    func performLSA(_ w: WeightedMatrix, components k: Int = 2) -> SVDResult {
        let docCount = w.documentCount
        let vocabCount = w.vocabularySize
        guard docCount > 0, vocabCount > 0 else {
            return SVDResult(coords: [], singularValues: [], spectrum: [],
                             labels: w.labels, categories: w.categories)
        }

        // 1. Gram matrix G = W · Wᵀ  (D × D). G[i][j] = dot(document i, document j).
        var g = [[Double]](repeating: [Double](repeating: 0, count: docCount), count: docCount)
        for i in 0..<docCount {
            let ri = w.rows[i]
            for j in i..<docCount {
                let rj = w.rows[j]
                var dot = 0.0
                for t in 0..<vocabCount { dot += ri[t] * rj[t] }
                g[i][j] = dot
                g[j][i] = dot          // symmetric
            }
        }

        // 2. Symmetric eigendecomposition  G = V · Λ · Vᵀ  (Jacobi rotations).
        let (eigenvalues, eigenvectors) = jacobiEigen(g)

        // 3. Order components by eigenvalue, largest first.
        let order = eigenvalues.indices.sorted { eigenvalues[$0] > eigenvalues[$1] }

        // Singular values σ_i = √(max(λ_i, 0)); full spectrum for the scree plot.
        let spectrum = order.map { sqrt(max(eigenvalues[$0], 0)) }

        // 4. Document coordinates = rows of U·Σ:  coord[d][c] = U[d, c] · σ_c,
        //    where U[d, c] is entry d of the c-th eigenvector (a column of V).
        let kept = min(k, docCount)
        var coords = [[Double]](repeating: [Double](repeating: 0, count: kept), count: docCount)
        var topSV = [Double](repeating: 0, count: kept)
        for c in 0..<kept {
            let col = order[c]
            let sigma = sqrt(max(eigenvalues[col], 0))
            topSV[c] = sigma
            for d in 0..<docCount {
                coords[d][c] = eigenvectors[d][col] * sigma
            }
        }

        return SVDResult(coords: coords, singularValues: topSV, spectrum: spectrum,
                         labels: w.labels, categories: w.categories)
    }

    /// Classic cyclic **Jacobi eigenvalue algorithm** for a symmetric matrix.
    ///
    /// It repeatedly applies 2×2 rotations Jᵀ·A·J that zero one off-diagonal
    /// entry at a time. Each rotation keeps A symmetric and nudges it toward a
    /// diagonal matrix; the accumulated rotations V become the eigenvectors and
    /// the final diagonal holds the eigenvalues.
    ///
    /// - Returns: (eigenvalues, eigenvectors) where `eigenvectors[i][k]` is the
    ///   i-th component of the k-th eigenvector (eigenvectors are columns).
    private func jacobiEigen(_ matrix: [[Double]]) -> (values: [Double], vectors: [[Double]]) {
        let n = matrix.count
        var a = matrix
        // Start with V = identity; it accumulates every rotation.
        var v = (0..<n).map { i in (0..<n).map { j in i == j ? 1.0 : 0.0 } }

        for _ in 0..<100 {                      // sweeps (plenty for n ≈ 100)
            // Stop once the off-diagonal mass is negligible.
            var off = 0.0
            for p in 0..<n { for q in (p + 1)..<n { off += a[p][q] * a[p][q] } }
            if off < 1e-18 { break }

            for p in 0..<(n - 1) {
                for q in (p + 1)..<n {
                    let apq = a[p][q]
                    if apq == 0 { continue }

                    // Rotation angle that zeros a[p][q] (smaller-angle solution).
                    let tau = (a[q][q] - a[p][p]) / (2 * apq)
                    let t = (tau >= 0 ? 1.0 : -1.0) / (abs(tau) + (tau * tau + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot()
                    let s = t * c

                    // A ← Jᵀ A J : rotate columns p,q, then rows p,q.
                    for i in 0..<n {
                        let aip = a[i][p], aiq = a[i][q]
                        a[i][p] = c * aip - s * aiq
                        a[i][q] = s * aip + c * aiq
                    }
                    for i in 0..<n {
                        let api = a[p][i], aqi = a[q][i]
                        a[p][i] = c * api - s * aqi
                        a[q][i] = s * api + c * aqi
                    }
                    // V ← V J : accumulate the rotation into the eigenvectors.
                    for i in 0..<n {
                        let vip = v[i][p], viq = v[i][q]
                        v[i][p] = c * vip - s * viq
                        v[i][q] = s * vip + c * viq
                    }
                }
            }
        }

        let values = (0..<n).map { a[$0][$0] }
        return (values, v)
    }
}
