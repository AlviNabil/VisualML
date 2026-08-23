//
//  NeuralActivationsView.swift
//  VisualML
//
//  Step 4 of the neural network walkthrough: the squash. Each activation is
//  shown as a curve, alongside its derivative -- which is the quantity
//  backpropagation actually multiplies by, and therefore what decides whether
//  a signal survives the trip back through a stack of layers.
//

import SwiftUI

struct NeuralActivationsView: View {
    let export: NeuralExport

    @State private var selected = 0
    @State private var showDerivative = true

    var body: some View {
        let activation = export.activations[selected]

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                picker

                ActivationCurveView(activation: activation,
                                    showDerivative: showDerivative)
                    .frame(height: 240)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                curveLegend
                Toggle("Show the derivative f′(z)", isOn: $showDerivative.animation())
                    .font(.subheadline)

                detail(activation)
                whyDerivativeMatters(activation)
                raceSection
            }
            .padding()
        }
        .navigationTitle("4 · Activations")
        .navigationBarTitleDisplayMode(.inline)
        .neuralInfo(export, topic: .activations)
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("The part that bends").font(.headline)
            Text("Between every pair of layers sits a function applied to each "
                 + "value on its own. Without it, stacking layers would achieve "
                 + "nothing. Which one is chosen changes how fast — and whether "
                 + "— the network learns.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var picker: some View {
        Picker("Activation", selection: $selected) {
            ForEach(export.activations.indices, id: \.self) { index in
                Text(export.activations[index].displayName).tag(index)
            }
        }
        .pickerStyle(.segmented)
    }

    private var curveLegend: some View {
        HStack(spacing: 16) {
            Label { Text("f(z)").font(.caption2) } icon: {
                Rectangle().fill(Color.accentColor).frame(width: 14, height: 2.5)
            }
            if showDerivative {
                Label { Text("f′(z)").font(.caption2) } icon: {
                    Rectangle().fill(.orange).frame(width: 14, height: 2.5)
                }
            }
            Spacer()
        }
        .foregroundStyle(.secondary)
    }

    // MARK: - Detail on the chosen function

    private func detail(_ activation: ActivationSample) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(activation.displayName).font(.headline)
            formula(activation.formula)
            formula(activation.derivative)
            HStack(spacing: 10) {
                stat("output range", activation.range)
                stat("steepest slope", String(format: "%.2f", activation.maxSlope))
            }
            Text(activation.note).font(.caption).foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func whyDerivativeMatters(_ activation: ActivationSample) -> some View {
        let layers = export.architecture.hiddenSizes.count
        let shrink = pow(activation.maxSlope, Double(layers))
        return VStack(alignment: .leading, spacing: 8) {
            Text("Why the slope decides everything").font(.headline)
            Text("Backpropagation multiplies the error by f′(z) once per layer "
                 + "on the way back. Multiply by a number smaller than 1 enough "
                 + "times and there is nothing left to learn from.")
                .font(.caption).foregroundStyle(.secondary)
            Text(String(format: "Best case here: %.2f steepest slope, across %d "
                        + "hidden layers, leaves at most %.2f of the signal — and "
                        + "that is only for units sitting at the steepest point.",
                        activation.maxSlope, layers, shrink))
                .font(.caption2.monospaced())
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    // MARK: - The race

    private var raceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Same network, different squash").font(.headline)
            Text("Identical architecture, identical starting weights, identical "
                 + "learning rate. The only change is the hidden activation.")
                .font(.caption).foregroundStyle(.secondary)

            ActivationRaceView(activations: export.activations, highlight: selected)
                .frame(height: 190)
                .padding(8)
                .background(Color.gray.opacity(0.10))
                .clipShape(RoundedRectangle(cornerRadius: 10))

            VStack(spacing: 4) {
                raceHeader
                ForEach(export.activations.indices, id: \.self) { index in
                    raceRow(index: index)
                }
            }
            .padding(10)
            .background(Color.gray.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            Text("Sigmoid is the outlier. Its slope never exceeds 0.25, so two "
                 + "hidden layers cut the returning signal to at most a sixteenth "
                 + "before it reaches the first layer. It sits at chance level for "
                 + "hundreds of epochs before breaking out — the vanishing "
                 + "gradient, in one picture.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var raceHeader: some View {
        HStack {
            Text("activation").font(.caption2.bold())
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("max f′").font(.caption2.bold()).frame(width: 52, alignment: .trailing)
            Text("epochs to 95%").font(.caption2.bold())
                .frame(width: 96, alignment: .trailing)
        }
        .foregroundStyle(.secondary)
    }

    private func raceRow(index: Int) -> some View {
        let activation = export.activations[index]
        let epochs = activation.training.epochsTo95
        let slowest = export.activations.compactMap(\.training.epochsTo95).max() ?? 1
        let isSelected = index == selected
        return HStack {
            HStack(spacing: 6) {
                Circle().fill(raceColour(index)).frame(width: 8, height: 8)
                Text(activation.displayName)
                    .font(.caption)
                    .fontWeight(isSelected ? .bold : .regular)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(String(format: "%.2f", activation.maxSlope))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .frame(width: 52, alignment: .trailing)
            HStack(spacing: 6) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.gray.opacity(0.15))
                        Capsule().fill(raceColour(index).opacity(0.65))
                            .frame(width: geo.size.width
                                   * CGFloat(Double(epochs ?? slowest) / Double(slowest)))
                    }
                }
                .frame(height: 8)
                Text(epochs.map(String.init) ?? "—")
                    .font(.caption2.monospacedDigit())
                    .frame(width: 34, alignment: .trailing)
            }
            .frame(width: 96)
        }
        .padding(.vertical, 2)
    }

    // MARK: - Pieces

    private func formula(_ text: String) -> some View {
        Text(text)
            .font(.caption.monospaced())
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// A stable colour per activation, shared by the chart and the table.
func raceColour(_ index: Int) -> Color {
    let palette: [Color] = [.red, .blue, .green, .orange, .purple, .teal]
    return palette[index % palette.count]
}

// MARK: - One activation, drawn

/// f(z) and optionally f'(z), over the sampled range.
struct ActivationCurveView: View {
    let activation: ActivationSample
    var showDerivative: Bool = true

    var body: some View {
        Canvas { context, size in
            let xs = activation.x
            guard let xLo = xs.min(), let xHi = xs.max() else { return }
            let pad: CGFloat = 26

            var values = activation.y
            if showDerivative { values += activation.dy }
            let yLo = Swift.min(values.min() ?? -1, -0.2)
            let yHi = Swift.max(values.max() ?? 1, 1.1)

            func sx(_ v: Double) -> CGFloat {
                pad + CGFloat((v - xLo) / (xHi - xLo)) * (size.width - 2 * pad)
            }
            func sy(_ v: Double) -> CGFloat {
                size.height - pad - CGFloat((v - yLo) / (yHi - yLo)) * (size.height - 2 * pad)
            }

            // Axes through the origin.
            var axes = Path()
            axes.move(to: CGPoint(x: pad, y: sy(0)))
            axes.addLine(to: CGPoint(x: size.width - pad, y: sy(0)))
            axes.move(to: CGPoint(x: sx(0), y: pad))
            axes.addLine(to: CGPoint(x: sx(0), y: size.height - pad))
            context.stroke(axes, with: .color(.gray.opacity(0.35)), lineWidth: 1)

            func curve(_ ys: [Double], colour: Color, width: CGFloat) {
                var path = Path()
                for (index, x) in xs.enumerated() {
                    let point = CGPoint(x: sx(x), y: sy(ys[index]))
                    if index == 0 { path.move(to: point) } else { path.addLine(to: point) }
                }
                context.stroke(path, with: .color(colour), lineWidth: width)
            }

            if showDerivative { curve(activation.dy, colour: .orange, width: 2) }
            curve(activation.y, colour: .accentColor, width: 2.5)

            context.draw(Text("z").font(.caption2).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width - pad + 8, y: sy(0)))
            context.draw(Text(String(format: "%.1f", yHi)).font(.system(size: 8))
                            .foregroundStyle(.secondary),
                         at: CGPoint(x: sx(0) - 12, y: pad), anchor: .trailing)
        }
    }
}

// MARK: - The race chart

/// Every activation's training loss on one set of axes.
struct ActivationRaceView: View {
    let activations: [ActivationSample]
    let highlight: Int

    var body: some View {
        Canvas { context, size in
            let allLosses = activations.flatMap { $0.training.curve.map(\.loss) }
            guard let lo = allLosses.min(), let hi = allLosses.max(), hi > lo else { return }
            let maxEpoch = Double(activations.first?.training.curve.last?.epoch ?? 1)
            let pad: CGFloat = 22

            // Log scales on both axes: the interesting behaviour is early and
            // the losses span several orders of magnitude.
            func sx(_ epoch: Double) -> CGFloat {
                let t = log10(epoch + 1) / log10(maxEpoch + 1)
                return pad + CGFloat(t) * (size.width - 2 * pad)
            }
            func sy(_ loss: Double) -> CGFloat {
                let t = (log10(Swift.max(loss, 1e-6)) - log10(Swift.max(lo, 1e-6)))
                    / (log10(hi) - log10(Swift.max(lo, 1e-6)))
                return size.height - pad - CGFloat(t) * (size.height - 2 * pad)
            }

            // Chance level: what a model that has learned nothing scores.
            let chance = log(2.0)
            if chance >= lo, chance <= hi {
                var line = Path()
                line.move(to: CGPoint(x: pad, y: sy(chance)))
                line.addLine(to: CGPoint(x: size.width - pad, y: sy(chance)))
                context.stroke(line, with: .color(.gray.opacity(0.55)),
                               style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                context.draw(Text("chance").font(.system(size: 8))
                                .foregroundStyle(.secondary),
                             at: CGPoint(x: size.width - pad - 2, y: sy(chance) - 7),
                             anchor: .trailing)
            }

            for (index, activation) in activations.enumerated() {
                var path = Path()
                for (position, point) in activation.training.curve.enumerated() {
                    let screen = CGPoint(x: sx(Double(point.epoch)), y: sy(point.loss))
                    if position == 0 { path.move(to: screen) } else { path.addLine(to: screen) }
                }
                let chosen = index == highlight
                context.stroke(path,
                               with: .color(raceColour(index).opacity(chosen ? 1 : 0.35)),
                               lineWidth: chosen ? 2.5 : 1.2)
            }

            context.draw(Text("epoch →").font(.system(size: 8)).foregroundStyle(.secondary),
                         at: CGPoint(x: size.width / 2, y: size.height - 6))
            context.draw(Text("loss").font(.system(size: 8)).foregroundStyle(.secondary),
                         at: CGPoint(x: 4, y: 8), anchor: .topLeading)
        }
    }
}
