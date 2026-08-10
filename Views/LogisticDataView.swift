//
//  LogisticDataView.swift
//  VisualML
//
//  Step 1 of the logistic walkthrough: what the data looks like when the target
//  is a yes/no, what goes wrong if a straight line is fitted to it, and the
//  shape the observed pass rate already traces.
//

import SwiftUI

struct LogisticDataView: View {
    let export: LogisticExport

    @State private var showLine = false
    @State private var showRates = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro
                BinaryScatterView(export: export,
                                  showLine: showLine, showRates: showRates)
                    .frame(height: 260)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                legend
                controls
                if showLine { lineProblem }
                if showRates { rateTable }
                counts
            }
            .padding()
        }
        .navigationTitle("1 · The data")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Copy

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("A yes/no target").font(.headline)
            Text("Every student sits at one of two heights: 0 for fail, 1 for pass. "
                 + "There is no middle. The question is what to predict for a value "
                 + "of hours we have not seen.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var legend: some View {
        HStack(spacing: 16) {
            Label { Text(export.dataset.classNames[1]).font(.caption) } icon: {
                Circle().fill(.green).frame(width: 9, height: 9)
            }
            Label { Text(export.dataset.classNames[0]).font(.caption) } icon: {
                Circle().fill(.red).frame(width: 9, height: 9)
            }
            Spacer()
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Try a straight line", isOn: $showLine.animation())
                .font(.subheadline)
            Toggle("Show the observed pass rate", isOn: $showRates.animation())
                .font(.subheadline)
        }
    }

    // MARK: - Why the line fails

    private var lineProblem: some View {
        let b = export.linearBaseline
        return VStack(alignment: .leading, spacing: 8) {
            Text("What goes wrong").font(.headline)
            Text(String(format: "Least squares gives p = %.4f × hours %@ %.4f. "
                        + "Nothing in that formula keeps the answer between 0 and 1.",
                        b.slope, b.intercept < 0 ? "−" : "+", abs(b.intercept)))
                .font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 10) {
                stat("lowest", String(format: "%.2f", b.minPrediction), .red)
                stat("highest", String(format: "%.2f", b.maxPrediction), .red)
                stat("outside 0–1", "\(b.outOfRangeCount)", .orange)
            }
            Text("A probability below 0 or above 1 has no meaning, so the line is "
                 + "the wrong shape for this job — it needs to bend and flatten at "
                 + "both ends.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.red.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Observed rates

    private var rateTable: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("The shape hiding in the data").font(.headline)
            Text("Group the students into bands of hours and take the share who "
                 + "passed in each. Nothing is fitted yet — this is just counting — "
                 + "and it already sweeps from near 0 up to near 1.")
                .font(.caption).foregroundStyle(.secondary)
            VStack(spacing: 3) {
                ForEach(export.empiricalRate) { band in
                    HStack(spacing: 8) {
                        Text(String(format: "%.1f–%.1fh", band.binStart, band.binEnd))
                            .font(.caption2.monospacedDigit())
                            .frame(width: 74, alignment: .leading)
                            .foregroundStyle(.secondary)
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(Color.gray.opacity(0.15))
                                Capsule().fill(Color.green.opacity(0.65))
                                    .frame(width: geo.size.width * band.rate)
                            }
                        }
                        .frame(height: 12)
                        Text("\(band.passed)/\(band.count)")
                            .font(.caption2.monospacedDigit())
                            .frame(width: 46, alignment: .trailing)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var counts: some View {
        let d = export.dataset
        return HStack(spacing: 10) {
            stat(d.classNames[1].lowercased(), "\(d.positiveCount)", .green)
            stat(d.classNames[0].lowercased(), "\(d.negativeCount)", .red)
            stat("students", "\(d.n)", .secondary)
        }
    }

    private func stat(_ title: String, _ value: String, _ tint: Color) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.headline.monospacedDigit()).foregroundStyle(tint)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - The scatter

/// Draws the 0/1 outcomes, optionally with the least-squares line and the
/// observed pass rate per band.
struct BinaryScatterView: View {
    let export: LogisticExport
    var showLine: Bool = false
    var showRates: Bool = false

    var body: some View {
        Canvas { context, size in
            let d = export.dataset
            let pad: CGFloat = 26
            let minX = d.mins[0], maxX = d.maxs[0]

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - minX) / max(maxX - minX, 1e-9)) * (size.width - 2 * pad)
            }
            // The vertical axis runs a little past 0 and 1 so the line can be
            // seen leaving the valid range.
            func sy(_ p: Double) -> CGFloat {
                let lo = -0.25, hi = 1.25
                return size.height - pad
                    - CGFloat((p - lo) / (hi - lo)) * (size.height - 2 * pad)
            }

            // Shade the impossible regions: a probability cannot live here.
            for band in [(1.0, 1.25), (-0.25, 0.0)] {
                let rect = CGRect(x: pad, y: sy(band.1),
                                  width: size.width - 2 * pad,
                                  height: sy(band.0) - sy(band.1))
                context.fill(Path(rect), with: .color(.red.opacity(0.06)))
            }

            // The two outcome levels.
            for level in [0.0, 1.0] {
                var line = Path()
                line.move(to: CGPoint(x: pad, y: sy(level)))
                line.addLine(to: CGPoint(x: size.width - pad, y: sy(level)))
                context.stroke(line, with: .color(.gray.opacity(0.35)),
                               style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }

            // Students, nudged apart vertically so overlaps stay countable. The
            // offset comes from the index, so it never changes between redraws.
            for (i, p) in d.points.enumerated() {
                let wobble = (Double((i * 37) % 11) / 11.0 - 0.5) * 0.16
                let c = CGPoint(x: sx(p.x[0]), y: sy(Double(p.y) + wobble))
                let r: CGFloat = 3
                context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r,
                                                    width: 2 * r, height: 2 * r)),
                             with: .color((p.y == 1 ? Color.green : .red).opacity(0.6)))
            }

            // Observed pass rate per band.
            if showRates {
                var path = Path()
                for (i, band) in export.empiricalRate.enumerated() {
                    let pt = CGPoint(x: sx(band.center), y: sy(band.rate))
                    if i == 0 { path.move(to: pt) } else { path.addLine(to: pt) }
                }
                context.stroke(path, with: .color(.blue.opacity(0.8)), lineWidth: 2)
                for band in export.empiricalRate {
                    let c = CGPoint(x: sx(band.center), y: sy(band.rate))
                    context.fill(Path(ellipseIn: CGRect(x: c.x - 4, y: c.y - 4,
                                                        width: 8, height: 8)),
                                 with: .color(.blue))
                }
            }

            // The least-squares line, which leaves the valid range at both ends.
            if showLine {
                let b = export.linearBaseline
                var path = Path()
                path.move(to: CGPoint(x: sx(minX), y: sy(b.predict(minX))))
                path.addLine(to: CGPoint(x: sx(maxX), y: sy(b.predict(maxX))))
                context.stroke(path, with: .color(.orange), lineWidth: 2.5)
            }

            context.draw(Text("1  \(export.dataset.classNames[1])").font(.caption2)
                            .foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: sy(1)), anchor: .leading)
            context.draw(Text("0  \(export.dataset.classNames[0])").font(.caption2)
                            .foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: sy(0)), anchor: .leading)
            context.draw(Text(export.dataset.axisLabels[0]).font(.caption2)
                            .foregroundStyle(.secondary),
                         at: CGPoint(x: size.width / 2, y: size.height - 6),
                         anchor: .bottom)
        }
    }
}
