//
//  DataPoint.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import Foundation

/// One row of our dataset: a single piece of text and the class it belongs to.
///
/// We keep TWO representations of the class on purpose:
///   - `category`: the human-readable name ("sport", "business") — for display.
///   - `label`:    the numeric class index (0, 1)            — for the math/ML.
///
/// `Identifiable` (a protocol that just requires an `id`) lets SwiftUI's `List`
/// and `ForEach` tell rows apart efficiently. `id = UUID()` gives every
/// DataPoint a unique identity automatically.
struct DataPoint: Identifiable {
    let id = UUID()
    let text: String
    let category: String
    let label: Int
}
