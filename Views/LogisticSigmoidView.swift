//
//  LogisticSigmoidView.swift
//  VisualML
//
//  Step 2 of the logistic walkthrough: the transformation. One feature value is
//  carried through both halves of the model — first the linear score z, then the
//  squash p = sigma(z) — with each stage drawn and the arithmetic spelled out.
//

import SwiftUI

struct LogisticSigmoidView: View {
    let export: LogisticExport

    /// The feature value being traced through the model.
    @State private var probe: Double = 6.5

    var body: some View {
        let fit = export.fit
        let z = fit.weight * probe + fit.intercept
        let p = sigmoid(z)

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                stageCard(number: "1", title: "Score the student",
                          formula: String(format: "z = %.4f × %.2f %@ %.4f = %+.3f",
                                          fit.weight, probe,
                                          fit.intercept < 0 ? "−" : "+",
                                          abs(fit.intercept), z)) {
                    ScoreLineView(export: export, probe: probe)
                        .frame(height: 130)
                }

                stageCard(number: "2", title: "Squash it into a probability",
                          formula: String(format: "p = σ(%+.3f) = 1 / (1 + e^%+.3f) = %.3f",
                                          z, -z, p)) {
                    SigmoidShapeView(z: z, p: p)
                        .frame(height: 150)
                }

                stageCard(number: "3", title: "Read it off the curve",
                          formula: String(format: "%.2f hours → %.1f%% chance of %@",
                                          probe, p * 100,
                                          export.dataset.classNames[1].lowercased())) {
                    ProbabilityCurveView(export: export, probe: probe)
                        .frame(height: 210)
                }

                probeControl
                verdict(p: p)
                whyThisShape
            }
            .padding()
        }
        .navigationTitle("2 · The transformation")
        .navigationBarTitleDisplayMode(.inline)
        .logisticInfo(export, topic: .model)
    }

    // MARK: - Copy

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Two steps, not one").font(.headline)
            Text("Logistic regression keeps the same straight-line score as before "
                 + "and then bends it. Drag the slider to carry one student through "
                 + "both halves.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var probeControl: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Hours studied").font(.caption)
                Spacer()
                Text(String(format: "%.2f", probe))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: $probe,
                   in: export.dataset.mins[0]...export.dataset.maxs[0])
        }
    }

    private func verdict(p: Double) -> some View {
        let names = export.dataset.classNames
        let predicted = p >= 0.5 ? names[1] : names[0]
        let tint: Color = p >= 0.5 ? .green : .red
        return HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Prediction").font(.caption2).foregroundStyle(.secondary)
                Text(predicted).font(.headline).foregroundStyle(tint)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("Confidence").font(.caption2).foregroundStyle(.secondary)
                Text(String(format: "%.1f%%", max(p, 1 - p) * 100))
                    .font(.headline.monospacedDigit())
            }
        }
        .padding(14)
        .background(tint.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var whyThisShape: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Why this shape").font(.headline)
            bullet("σ never reaches 0 or 1, so every prediction is a usable probability.")
            bullet("It is steepest in the middle, where the evidence is genuinely balanced.")
            bullet("It flattens at both ends: once a case is clear, more hours barely move it.")
            bullet(String(format: "p crosses 0.5 exactly where z = 0, at %.2f hours.",
                          export.fit.boundary))
        }
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").font(.caption).foregroundStyle(.secondary)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - Card shell

    private func stageCard<Content: View>(number: String, title: String,
                                          formula: String,
                                          @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text(number)
                    .font(.caption.bold())
                    .frame(width: 20, height: 20)
                    .background(Color.accentColor, in: Circle())
                    .foregroundStyle(.white)
                Text(title).font(.subheadline.bold())
            }
            content()
            Text(formula)
                .font(.caption.monospaced())
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(12)
        .background(Color.gray.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Stage 1: the straight-line score

/// The unbounded score z against the feature, with the probe marked.
struct ScoreLineView: View {
    let export: LogisticExport
    let probe: Double

    var body: some View {
        Canvas { context, size in
            let curve = export.curve
            guard let zLo = curve.map(\.z).min(), let zHi = curve.map(\.z).max() else { return }
            let xLo = curve.first!.x, xHi = curve.last!.x
            let pad: CGFloat = 18

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - xLo) / (xHi - xLo)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat((v - zLo) / (zHi - zLo)) * (size.height - 2 * pad)
            }

            // z = 0 is the level that will become p = 0.5.
            var zero = Path()
            zero.move(to: CGPoint(x: pad, y: sy(0)))
            zero.addLine(to: CGPoint(x: size.width - pad, y: sy(0)))
            context.stroke(zero, with: .color(.gray.opacity(0.5)),
                           style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
            context.draw(Text("z = 0").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width - pad - 2, y: sy(0) - 8), anchor: .trailing)

            var line = Path()
            line.move(to: CGPoint(x: sx(xLo), y: sy(curve.first!.z)))
            line.addLine(to: CGPoint(x: sx(xHi), y: sy(curve.last!.z)))
            context.stroke(line, with: .color(.purple), lineWidth: 2.5)

            let zProbe = export.fit.weight * probe + export.fit.intercept
            markProbe(context: context, size: size,
                      at: CGPoint(x: sx(probe), y: sy(zProbe)), tint: .purple)
        }
    }
}

// MARK: - Stage 2: the sigmoid itself

/// The sigmoid function drawn in its own coordinates, z across and p up.
struct SigmoidShapeView: View {
    let z: Double
    let p: Double

    var body: some View {
        Canvas { context, size in
            let pad: CGFloat = 18
            let span = max(6.0, abs(z) + 1.5)     // always keep the probe visible

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v + span) / (2 * span)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat(v) * (size.height - 2 * pad)
            }

            for level in [0.0, 0.5, 1.0] {
                var grid = Path()
                grid.move(to: CGPoint(x: pad, y: sy(level)))
                grid.addLine(to: CGPoint(x: size.width - pad, y: sy(level)))
                context.stroke(grid, with: .color(.gray.opacity(level == 0.5 ? 0.45 : 0.22)),
                               style: StrokeStyle(lineWidth: 1, dash: level == 0.5 ? [4, 3] : []))
            }

            var curve = Path()
            for i in 0...160 {
                let t = -span + 2 * span * Double(i) / 160
                let point = CGPoint(x: sx(t), y: sy(sigmoid(t)))
                if i == 0 { curve.move(to: point) } else { curve.addLine(to: point) }
            }
            context.stroke(curve, with: .color(.blue), lineWidth: 2.5)

            markProbe(context: context, size: size,
                      at: CGPoint(x: sx(z), y: sy(p)), tint: .blue)

            context.draw(Text("σ(z)").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: pad + 2, y: pad), anchor: .topLeading)
            context.draw(Text("z").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width - pad, y: size.height - 4), anchor: .bottomTrailing)
        }
    }
}

// MARK: - Stage 3: the curve over the real data

/// The fitted probability curve drawn over the students, with the boundary.
struct ProbabilityCurveView: View {
    let export: LogisticExport
    let probe: Double

    var body: some View {
        Canvas { context, size in
            let d = export.dataset
            let curve = export.curve
            let xLo = curve.first!.x, xHi = curve.last!.x
            let pad: CGFloat = 22

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - xLo) / (xHi - xLo)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat(v) * (size.height - 2 * pad)
            }

            for level in [0.0, 0.5, 1.0] {
                var grid = Path()
                grid.move(to: CGPoint(x: pad, y: sy(level)))
                grid.addLine(to: CGPoint(x: size.width - pad, y: sy(level)))
                context.stroke(grid, with: .color(.gray.opacity(level == 0.5 ? 0.45 : 0.22)),
                               style: StrokeStyle(lineWidth: 1, dash: level == 0.5 ? [4, 3] : []))
            }

            // The students, sitting on the 0 and 1 lines.
            for (i, point) in d.points.enumerated() {
                let wobble = (Double((i * 37) % 11) / 11.0 - 0.5) * 0.10
                let c = CGPoint(x: sx(point.x[0]), y: sy(Double(point.y) + wobble))
                context.fill(Path(ellipseIn: CGRect(x: c.x - 2.5, y: c.y - 2.5,
                                                    width: 5, height: 5)),
                             with: .color((point.y == 1 ? Color.green : .red).opacity(0.45)))
            }

            // The decision boundary, where the curve crosses one half.
            var boundary = Path()
            boundary.move(to: CGPoint(x: sx(export.fit.boundary), y: pad))
            boundary.addLine(to: CGPoint(x: sx(export.fit.boundary), y: size.height - pad))
            context.stroke(boundary, with: .color(.primary.opacity(0.45)),
                           style: StrokeStyle(lineWidth: 1.5, dash: [5, 4]))

            var path = Path()
            for (i, sample) in curve.enumerated() {
                let point = CGPoint(x: sx(sample.x), y: sy(sample.p))
                if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
            }
            context.stroke(path, with: .color(.blue), lineWidth: 3)

            let p = sigmoid(export.fit.weight * probe + export.fit.intercept)
            markProbe(context: context, size: size,
                      at: CGPoint(x: sx(probe), y: sy(p)), tint: .blue)

            context.draw(Text("1.0").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: sy(1)), anchor: .leading)
            context.draw(Text("0.5").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: sy(0.5)), anchor: .leading)
            context.draw(Text("0.0").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: sy(0)), anchor: .leading)
        }
    }
}

// MARK: - Shared marker

/// Draws the probe: a crosshair down to the axis plus a filled dot.
private func markProbe(context: GraphicsContext, size: CGSize,
                       at point: CGPoint, tint: Color) {
    var guides = Path()
    guides.move(to: CGPoint(x: point.x, y: size.height))
    guides.addLine(to: point)
    guides.addLine(to: CGPoint(x: 0, y: point.y))
    context.stroke(guides, with: .color(tint.opacity(0.35)),
                   style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
    context.fill(Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5,
                                        width: 10, height: 10)),
                 with: .color(tint))
    context.stroke(Path(ellipseIn: CGRect(x: point.x - 7.5, y: point.y - 7.5,
                                          width: 15, height: 15)),
                   with: .color(.white.opacity(0.9)), lineWidth: 1.5)
}
