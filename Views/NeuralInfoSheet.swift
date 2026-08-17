//
//  NeuralInfoSheet.swift
//  VisualML
//
//  The maths behind the neural network stage, worked through with the numbers
//  this app shipped. Topics are added as the walkthrough steps land; each step
//  opens the sheet on its own section.
//

import SwiftUI

struct NeuralInfoSheet: View {
    let export: NeuralExport
    var initialTopic: Topic = .setup

    @Environment(\.dismiss) private var dismiss
    @State private var topic: Topic?

    enum Topic: String, CaseIterable, Identifiable {
        case setup = "Setup"
        case architecture = "Layers"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Topic", selection: Binding(
                    get: { topic ?? initialTopic },
                    set: { topic = $0 })) {
                    ForEach(Topic.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding()

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        switch topic ?? initialTopic {
                        case .setup: setupSection
                        case .architecture: architectureSection
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 28)
                }
            }
            .navigationTitle("The maths")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    // MARK: - Setup

    private var setupSection: some View {
        let d = export.dataset
        return VStack(alignment: .leading, spacing: 14) {
            heading("A problem with a curved answer")
            body("\(d.n) points on a plane. One class fills a disc at the centre, "
                 + "the other forms a ring around it. They never overlap, so the "
                 + "problem is perfectly solvable — the difficulty is the shape of "
                 + "the answer, not the data.")

            heading("Why every earlier model fails")
            body("Logistic regression, the SVM, the LSA classifier — each ends in "
                 + "a single expression of the form:")
            formula("decide class 1 when   w·x + b ≥ 0")
            body("That inequality describes a half-plane: everything on one side "
                 + "of a straight cut. A disc surrounded by a ring cannot be "
                 + "carved out by one straight cut at any angle.")
            note(String(format: "Measured on this data: a fitted straight "
                        + "boundary reaches %.1f%%, against 50%% for a coin flip. "
                        + "The network reaches %.1f%%.",
                        d.linearBaselineAccuracy * 100,
                        export.training.finalAccuracy * 100))

            heading("What actually separates these classes")
            body("Distance from the centre. A model given x₁² + x₂² as a feature "
                 + "would solve this instantly with a straight cut in that new "
                 + "space. The interesting part is that a neural network is never "
                 + "told to square anything — it has to construct something with "
                 + "that effect out of weighted sums and squashes.")
        }
    }

    // MARK: - Architecture

    private var architectureSection: some View {
        let a = export.architecture
        return VStack(alignment: .leading, spacing: 14) {
            heading("One neuron is a logistic regression")
            body("A single unit takes the values arriving from the layer below, "
                 + "weights them, adds a bias, and squashes the result:")
            formula("z = w·x + b\na = f(z)")
            body("That is exactly the logistic regression from the earlier stage, "
                 + "with f as the sigmoid. Stacking these is the only new idea.")

            heading("A layer, in matrix form")
            body("Rather than one unit at a time, a whole layer computes at once. "
                 + "For a batch of n rows arriving at a layer with d units:")
            formula("Z = A·W + b        (n × d)\nA' = f(Z)          (n × d)")
            body("W holds one column per unit in the layer, one row per input. "
                 + "This network's three weight matrices are:")
            matrixShapes

            heading("Why the squash is not optional")
            body("If f were removed — if each layer were just a weighted sum — "
                 + "then two layers in a row would be:")
            formula("(X·W₁ + b₁)·W₂ + b₂  =  X·(W₁W₂) + (b₁W₂ + b₂)")
            note("W₁W₂ is just another matrix. A stack of purely linear layers "
                 + "collapses algebraically into one linear layer, so it could "
                 + "still only draw a straight boundary — no matter how many "
                 + "layers were added. The non-linear f between layers is the "
                 + "entire reason depth buys anything.")

            heading("The size of this one")
            body(String(format: "%d parameters in total, at a learning rate of "
                        + "%g for %d epochs. Hidden layers use %@; the output "
                        + "uses %@ so the result reads as a probability.",
                        a.parameterCount, a.learningRate, a.epochs,
                        a.hiddenActivation, a.outputActivation))
        }
    }

    private var matrixShapes: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(export.architecture.weightShapes.enumerated()), id: \.offset) { index, shape in
                Text("W\(index + 1): \(shape.0) × \(shape.1)   b\(index + 1): \(shape.1)")
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.gray.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Building blocks

    private func heading(_ text: String) -> some View {
        Text(text).font(.headline).padding(.top, 4)
    }

    private func body(_ text: String) -> some View {
        Text(text).font(.subheadline).foregroundStyle(.secondary)
    }

    private func formula(_ text: String) -> some View {
        Text(text)
            .font(.callout.monospaced())
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func note(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "lightbulb.fill")
                .font(.caption2).foregroundStyle(.yellow)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.yellow.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Reaching it from any step

struct NeuralInfoButton: ViewModifier {
    let export: NeuralExport
    let topic: NeuralInfoSheet.Topic
    @State private var showing = false

    func body(content: Content) -> some View {
        content
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showing = true } label: { Image(systemName: "info.circle") }
                        .accessibilityLabel("The maths behind this step")
                }
            }
            .sheet(isPresented: $showing) {
                NeuralInfoSheet(export: export, initialTopic: topic)
                    .presentationDetents([.large])
            }
    }
}

extension View {
    /// Attaches the neural-network maths sheet, opening on this step's section.
    func neuralInfo(_ export: NeuralExport, topic: NeuralInfoSheet.Topic) -> some View {
        modifier(NeuralInfoButton(export: export, topic: topic))
    }
}
