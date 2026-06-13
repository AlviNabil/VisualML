//
//  MatrixMath.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/13/26.
//

import Foundation
import Accelerate

class MatrixMath {
    
    func performLSA(flattenedMatrix: [Double], rows: Int, columns: Int) -> [(x: Double, y: Double)] {
        // 1. Prepare C-compatible variables
        var a = flattenedMatrix
        var m = __CLPK_integer(columns) // Columns (Words)
        var n = __CLPK_integer(rows)    // Rows (Documents)
        var lda = m
        
        // 2. Output matrices
        var u = [Double](repeating: 0.0, count: Int(m * m))
        var ldu = m
        
        var s = [Double](repeating: 0.0, count: Int(min(m, n)))
        
        var vt = [Double](repeating: 0.0, count: Int(n * n))
        var ldvt = n
        
        // 3. Configuration parameters for LAPACK
        var jobu: CChar = 83  // ASCII for 'S' (Compute first min(m,n) columns)
        var jobvt: CChar = 83 // ASCII for 'S'
        
        var lwork = __CLPK_integer(-1) // -1 triggers a "Workspace Query"
        var work = [Double](repeating: 0.0, count: 1)
        var info = __CLPK_integer(0)
        
        // 4. First Pass: Workspace Query
        dgesvd_(&jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        
        // 5. Second Pass: Actual Computation
        lwork = __CLPK_integer(work[0])
        work = [Double](repeating: 0.0, count: Int(lwork))
        
        dgesvd_(&jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        
        // 6. Extract 2D Coordinates (using VT matrix because of Fortran transposition)
        var coordinates: [(x: Double, y: Double)] = []
        for i in 0..<rows {
            // Scale the principal components by the singular values
            let x = vt[i * Int(n)] * s[0]
            let y = vt[i * Int(n) + 1] * s[1]
            coordinates.append((x: x, y: y))
        }
        
        return coordinates
    }
}
