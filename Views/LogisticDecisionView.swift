//
//  LogisticDecisionView.swift
//  VisualML
//
//  Step 4 of the logistic walkthrough: turning a probability into a decision.
//  The cut between the two classes is movable, and moving it trades one kind of
//  mistake for the other.
//

import SwiftUI

struct LogisticDecisionView: View {
    let export: LogisticExport

    /// Index into the precomputed threshold sweep.
    @State private var cutIndex: Int = 9      // 0.50
    @State private var showTest = false

    var body: some View {
        let cut = export.thresholdSweep[cutIndex]
        let confusion = showTest ? cut.testConfusion : cut.confusion
        let accuracy = showTest ? cut.testAccuracy : cut.accuracy
        let boundary = boundaryHours(for: cut.threshold)

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                DecisionRegionView(export: export, threshold: cut.threshold,
                                   boundary: boundary, showTest: showTest)
                    .frame(height: 240)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                boundaryReadout(cut: cut, boundary: boundary)
                thresholdControl(cut: cut)
                Picker("Set", selection: $showTest) {
                    Text("Training").tag(false)
                    Text("Held-out test").tag(true)
                }
                .pickerStyle(.segmented)

                confusionSection(confusion, accuracy: accuracy)
                tradeoff(cut)
                oddsSection
            }
            .padding()
        }
        .navigationTitle("4 · The decision")
        .navigationBarTitleDisplayMode(.inline)
    }

    /// The hours at which the model's probability reaches the chosen cut.
    /// Solving σ(w·x + b) = t for x gives x = (log(t/(1−t)) − b) / w.
    private func boundaryHours(for threshold: Double) -> Double {
        let fit = export.fit
        guard abs(fit.weight) > 1e-9 else { return .nan }
        return (log(threshold / (1 - threshold)) - fit.intercept) / fit.weight
    }

    // MARK: - Copy

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("From a probability to an answer").font(.headline)
            Text("The model outputs a chance, not a verdict. To decide, pick a cut: "
                 + "above it, predict pass. Where that cut sits is a choice, not "
                 + "something the maths settles.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func boundaryReadout(cut: ThresholdPoint, boundary: Double) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(String(format: "Predict %@ above %.2f hours",
                        export.dataset.classNames[1].lowercased(), boundary))
                .font(.callout.monospaced()).foregroundStyle(Color.accentColor)
            Text(String(format: "σ(%.4f × x %@ %.4f) = %.2f  ⟹  x = %.2f",
                        export.fit.weight,
                        export.fit.intercept < 0 ? "−" : "+", abs(export.fit.intercept),
                        cut.threshold, boundary))
                .font(.caption2.monospaced()).foregroundStyle(.secondary)
        }
    }

    private func thresholdControl(cut: ThresholdPoint) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Decision threshold").font(.caption)
                Spacer()
                Text(String(format: "p ≥ %.2f", cut.threshold))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: Binding(
                get: { Double(cutIndex) },
                set: { cutIndex = Int($0.rounded()) }),
                   in: 0...Double(export.thresholdSweep.count - 1), step: 1)
        }
    }

    // MARK: - Confusion

    private func confusionSection(_ c: [[Int]], accuracy: Double) -> some View {
        let names = export.dataset.classNames
        return VStack(alignment: .leading, spacing: 8) {
            Text("What it gets right and wrong").font(.headline)
            Grid(horizontalSpacing: 8, verticalSpacing: 6) {
                GridRow {
                    Text("").gridColumnAlignment(.leading)
                    Text("said \(names[0].lowercased())").font(.caption2).bold()
                    Text("said \(names[1].lowercased())").font(.caption2).bold()
                }
                GridRow {
                    Text("really \(names[0].lowercased())").font(.caption2).bold()
                    cell(c[0][0], correct: true)
                    cell(c[0][1], correct: false)
                }
                GridRow {
                    Text("really \(names[1].lowercased())").font(.caption2).bold()
                    cell(c[1][0], correct: false)
                    cell(c[1][1], correct: true)
                }
            }
            HStack(spacing: 10) {
                stat("accuracy", String(format: "%.1f%%", accuracy * 100))
                stat("always guess \(export.dataset.classNames[0].lowercased())",
                     String(format: "%.1f%%", export.fit.majorityAccuracy * 100))
            }
        }
    }

    private func cell(_ n: Int, correct: Bool) -> some View {
        Text("\(n)")
            .font(.callout.monospacedDigit()).bold()
            .frame(maxWidth: .infinity)
            .frame(height: 40)
            .background((correct ? Color.green : Color.red).opacity(n > 0 ? 0.18 : 0.06))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit())
            Text(title).font(.caption2).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - The trade

    private func tradeoff(_ cut: ThresholdPoint) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("The trade").font(.headline)
            bar("caught (recall)", cut.recall, .green)
            bar("right when it says pass (precision)", cut.precision, .blue)
            Text("Lower the cut and almost every passing student is caught, but "
                 + "many failing ones get flagged too. Raise it and the flags "
                 + "become trustworthy while more passes slip through. There is no "
                 + "setting that wins both.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func bar(_ label: String, _ value: Double, _ tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(label).font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.0f%%", value * 100))
                    .font(.caption2.monospacedDigit())
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.gray.opacity(0.18))
                    Capsule().fill(tint.opacity(0.7))
                        .frame(width: geo.size.width * value)
                }
            }
            .frame(height: 8)
        }
    }

    // MARK: - What w means

    private var oddsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("What the weight means").font(.headline)
            Text(String(format: "w = %.4f, and e^w = %.2f.", export.fit.weight,
                        export.fit.oddsRatio))
                .font(.callout.monospaced())
            Text(String(format: "Each extra hour multiplies the odds of passing by "
                        + "%.2f. Odds, not probability: at even odds one more hour "
                        + "moves the answer a lot, but at 95%% there is little room "
                        + "left to move.", export.fit.oddsRatio))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Decision regions

/// The curve with the two decision regions shaded, and every student shown as
/// correct or mistaken at the current cut.
struct DecisionRegionView: View {
    let export: LogisticExport
    let threshold: Double
    let boundary: Double
    var showTest: Bool = false

    var body: some View {
        Canvas { context, size in
            let xLo = export.curve.first!.x, xHi = export.curve.last!.x
            let pad: CGFloat = 22

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - xLo) / (xHi - xLo)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat(v) * (size.height - 2 * pad)
            }

            // Shade the two sides of the cut.
            if boundary.isFinite {
                let cutX = min(max(sx(boundary), pad), size.width - pad)
                context.fill(Path(CGRect(x: pad, y: pad, width: cutX - pad,
                                         height: size.height - 2 * pad)),
                             with: .color(.red.opacity(0.07)))
                context.fill(Path(CGRect(x: cutX, y: pad, width: size.width - pad - cutX,
                                         height: size.height - 2 * pad)),
                             with: .color(.green.opacity(0.07)))
            }

            // The threshold level and where the curve crosses it.
            var level = Path()
            level.move(to: CGPoint(x: pad, y: sy(threshold)))
            level.addLine(to: CGPoint(x: size.width - pad, y: sy(threshold)))
            context.stroke(level, with: .color(.primary.opacity(0.45)),
                           style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))

            if boundary.isFinite, boundary > xLo, boundary < xHi {
                var cut = Path()
                cut.move(to: CGPoint(x: sx(boundary), y: pad))
                cut.addLine(to: CGPoint(x: sx(boundary), y: size.height - pad))
                context.stroke(cut, with: .color(.primary.opacity(0.45)),
                               style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
            }

            // Students, marked by whether this cut classifies them correctly.
            for (i, point) in export.dataset.points.enumerated() where point.test == showTest {
                let p = sigmoid(export.fit.weight * point.x[0] + export.fit.intercept)
                let predicted = p >= threshold ? 1 : 0
                let right = predicted == point.y
                let wobble = (Double((i * 37) % 11) / 11.0 - 0.5) * 0.10
                let c = CGPoint(x: sx(point.x[0]), y: sy(Double(point.y) + wobble))
                let r: CGFloat = right ? 2.8 : 4
                context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r,
                                                    width: 2 * r, height: 2 * r)),
                             with: .color(right
                                          ? (point.y == 1 ? Color.green : .red).opacity(0.45)
                                          : .orange))
            }

            // The fitted curve.
            var path = Path()
            for (i, sample) in export.curve.enumerated() {
                let point = CGPoint(x: sx(sample.x), y: sy(sample.p))
                if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
            }
            context.stroke(path, with: .color(.blue), lineWidth: 3)

            context.draw(Text("mistakes in orange").font(.caption2)
                            .foregroundStyle(.orange),
                         at: CGPoint(x: size.width - pad, y: pad), anchor: .topTrailing)
        }
    }
}
