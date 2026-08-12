//
//  ClusterMapView.swift
//  VisualML
//
//  The shared scatter used by every k-means step: customers as points, coloured
//  by cluster once there is a cluster to colour by, with centroids marked as
//  large X's. Axis bounds are fixed to the dataset's full range so the same
//  view can be reused frame to frame without the scale jumping around.
//

import SwiftUI

/// A palette wide enough for the largest k the elbow curve tries (8).
let clusterPalette: [Color] = [
    .blue, .orange, .green, .purple, .pink, .cyan, .brown, .indigo
]

func clusterColor(_ index: Int) -> Color {
    clusterPalette[index % clusterPalette.count]
}

struct ClusterMapView: View {
    let dataset: ClusterDataset
    /// Cluster index per point, in the same order as `dataset.points`. `nil`
    /// draws every point in a single neutral colour.
    var assignments: [Int]? = nil
    /// Centroid positions in real feature units. `nil` draws no markers.
    var centroids: [[Double]]? = nil

    var body: some View {
        Canvas { context, size in
            let pad: CGFloat = 24
            let minX = dataset.mins[0], maxX = dataset.maxs[0]
            let minY = dataset.mins[1], maxY = dataset.maxs[1]

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - minX) / max(maxX - minX, 1e-9)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat((v - minY) / max(maxY - minY, 1e-9)) * (size.height - 2 * pad)
            }

            // Frame
            context.stroke(Path(CGRect(x: pad, y: pad, width: size.width - 2 * pad,
                                       height: size.height - 2 * pad)),
                           with: .color(.gray.opacity(0.25)), lineWidth: 1)

            // Customers
            for (i, point) in dataset.points.enumerated() {
                let c = CGPoint(x: sx(point.x[0]), y: sy(point.x[1]))
                let color = assignments.map { clusterColor($0[i]) } ?? Color.gray
                let r: CGFloat = 3.2
                context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                             with: .color(color.opacity(0.7)))
            }

            // Centroids: a dark X with a coloured fill behind it.
            if let centroids {
                for (i, centroid) in centroids.enumerated() {
                    let c = CGPoint(x: sx(centroid[0]), y: sy(centroid[1]))
                    let r: CGFloat = 8
                    context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                                 with: .color(clusterColor(i)))
                    var cross = Path()
                    cross.move(to: CGPoint(x: c.x - r * 0.6, y: c.y - r * 0.6))
                    cross.addLine(to: CGPoint(x: c.x + r * 0.6, y: c.y + r * 0.6))
                    cross.move(to: CGPoint(x: c.x + r * 0.6, y: c.y - r * 0.6))
                    cross.addLine(to: CGPoint(x: c.x - r * 0.6, y: c.y + r * 0.6))
                    context.stroke(cross, with: .color(.white), lineWidth: 2)
                    context.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                                   with: .color(.black.opacity(0.6)), lineWidth: 1.2)
                }
            }

            context.draw(Text(dataset.axisLabels[0]).font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width / 2, y: size.height - 4), anchor: .bottom)
            context.draw(Text(dataset.axisLabels[1]).font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: 8), anchor: .topLeading)
        }
    }
}
