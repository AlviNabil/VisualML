//
//  NeuralArchitectureView.swift
//  VisualML
//
//  Step 2 of the neural network walkthrough: the shape of the model. What the
//  layers are, how many numbers the network actually stores, and where those
//  numbers live.
//

import SwiftUI

struct NeuralArchitectureView: View {
    let export: NeuralExport

    var body: some View {
        let architecture = export.architecture
        let finalFrame = export.training.frames.last

        return ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                intro

                NetworkDiagramView(layerSizes: architecture.layerSizes,
                                   weights: finalFrame?.weights,
                                   showValues: false)
                    .frame(height: 240)
                    .padding(8)
                    .background(Color.gray.opacity(0.10))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                WeightLegend()
                layerTable
                parameterCount
                whyLayers
            }
            .padding()
        }
        .navigationTitle("2 · The architecture")
        .navigationBarTitleDisplayMode(.inline)
        .neuralInfo(export, topic: .architecture)
    }

    private var intro: some View {
        let a = export.architecture
        let shape = a.layerSizes.map(String.init).joined(separator: " → ")
        return VStack(alignment: .leading, spacing: 6) {
            Text("Layers of neurons").font(.headline)
            Text("This network is \(shape): two inputs, two hidden layers of "
                 + "\(a.hiddenSizes.first ?? 0), and one output. Small enough "
                 + "that every single number in it fits on this screen — which "
                 + "is the point.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: - The layers

    private var layerTable: some View {
        let a = export.architecture
        return VStack(alignment: .leading, spacing: 8) {
            Text("What each layer holds").font(.headline)
            VStack(spacing: 0) {
                header
                Divider()
                ForEach(Array(a.weightShapes.enumerated()), id: \.offset) { index, shape in
                    layerRow(index: index, shape: shape)
                    if index < a.weightShapes.count - 1 { Divider().opacity(0.4) }
                }
            }
            .padding(10)
            .background(Color.gray.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var header: some View {
        HStack {
            Text("layer").font(.caption2.bold()).frame(width: 50, alignment: .leading)
            Text("W shape").font(.caption2.bold()).frame(maxWidth: .infinity, alignment: .leading)
            Text("activation").font(.caption2.bold()).frame(maxWidth: .infinity, alignment: .leading)
            Text("params").font(.caption2.bold()).frame(width: 52, alignment: .trailing)
        }
        .foregroundStyle(.secondary)
        .padding(.bottom, 4)
    }

    private func layerRow(index: Int, shape: (Int, Int)) -> some View {
        let a = export.architecture
        let last = index == a.weightShapes.count - 1
        let params = shape.0 * shape.1 + shape.1
        return HStack {
            Text("\(index + 1)").font(.caption.bold())
                .frame(width: 50, alignment: .leading)
            Text("\(shape.0) × \(shape.1)")
                .font(.caption.monospacedDigit())
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(last ? a.outputActivation : a.hiddenActivation)
                .font(.caption.monospaced())
                .foregroundStyle(last ? Color.green : Color.accentColor)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("\(params)").font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 52, alignment: .trailing)
        }
        .padding(.vertical, 5)
    }

    private var parameterCount: some View {
        let a = export.architecture
        return VStack(alignment: .leading, spacing: 8) {
            Text("The whole model is \(a.parameterCount) numbers").font(.headline)
            Text("Every weight and every bias, added up. That is all that gets "
                 + "adjusted during training, and all that has to be kept "
                 + "afterwards to classify a new point. A production model has "
                 + "billions of these; the arithmetic is identical.")
                .font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 10) {
                stat("inputs", "\(a.inputCount)")
                stat("hidden units", "\(a.hiddenSizes.reduce(0, +))")
                stat("outputs", "\(a.outputCount)")
                stat("parameters", "\(a.parameterCount)")
            }
        }
    }

    private var whyLayers: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Why more than one layer").font(.headline)
            bullet("Each hidden unit computes exactly what logistic regression "
                   + "computes: a weighted sum, then a squash. On its own it can "
                   + "only draw a straight cut.")
            bullet("A layer of them draws several cuts at once, each at its own "
                   + "angle and offset.")
            bullet("The next layer combines those cuts, and combinations of "
                   + "straight cuts can enclose a region — which is how a closed "
                   + "loop around the middle class becomes reachable.")
            Text("Without the activation function in between, though, all of "
                 + "this collapses: a stack of purely linear layers multiplies "
                 + "out to a single linear layer, no matter how deep it is. The "
                 + "squash is what makes depth mean anything.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").font(.caption).foregroundStyle(.secondary)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
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
}
