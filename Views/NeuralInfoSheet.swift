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
        case forward = "Forward"
        case activations = "Activations"
        case backprop = "Backprop"
        case training = "Training"
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
                        case .forward: forwardSection
                        case .activations: activationsSection
                        case .backprop: backpropSection
                        case .training: trainingSection
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

    // MARK: - Forward

    private var forwardSection: some View {
        let trace = export.forwardTrace
        return VStack(alignment: .leading, spacing: 14) {
            heading("Inference is three lines of arithmetic")
            body("Running the network on a point is nothing more than repeating "
                 + "the same two operations once per layer:")
            formula("z = a·W + b\na = f(z)")
            body("The output of one layer becomes the input of the next, and the "
                 + "last layer's activation is the answer.")

            heading("Followed end to end")
            body(String(format: "For the point (%.3f, %.3f):",
                        trace.input[0], trace.input[1]))
            formula(chainSummary(trace))
            note(String(format: "The final number, %.4f, is the probability the "
                        + "network assigns to the second class. Its true class is "
                        + "%d, so this one is right.",
                        trace.prediction, trace.label))

            heading("Where the shapes come from")
            body("Each weight matrix has one row per incoming value and one "
                 + "column per unit in the layer, so multiplying by it maps a "
                 + "vector of one width to a vector of another. That is the only "
                 + "thing changing the number of values from layer to layer.")

            heading("A whole batch at once")
            body("Nothing above changes when many points are pushed through "
                 + "together — a matrix of n rows goes in, a matrix of n rows "
                 + "comes out, and the same weights are used for every row. That "
                 + "is why this arithmetic runs well on hardware built for "
                 + "matrix multiplication.")
        }
    }

    private func chainSummary(_ trace: ForwardTrace) -> String {
        var lines: [String] = ["x        (\(trace.input.count) values)"]
        for step in trace.steps {
            lines.append("layer \(step.layer)  \(step.input.count) → \(step.a.count)"
                         + "   \(step.activation)")
        }
        lines.append(String(format: "output   %.4f", trace.prediction))
        return lines.joined(separator: "\n")
    }

    // MARK: - Activations

    private var activationsSection: some View {
        let hiddenLayers = export.architecture.hiddenSizes.count
        let sigmoid = export.activations.first { $0.name == "sigmoid" }
        let relu = export.activations.first { $0.name == "relu" }
        return VStack(alignment: .leading, spacing: 14) {
            heading("What the squash is for")
            body("Applied to each value on its own, after the weighted sum. Its "
                 + "only job is to be non-linear — without it, any stack of "
                 + "layers collapses into a single linear one.")

            heading("Why its derivative is the thing that matters")
            body("Training sends an error signal backward through the layers, "
                 + "and at every layer that signal is multiplied by f′(z):")
            formula("dL/dz[l] = dL/da[l] ⊙ f′(z[l])")
            body("So the derivative is a volume knob on learning. Where f′ is "
                 + "near zero, almost nothing gets through, and the units in that "
                 + "layer barely move however wrong the answer was.")

            heading("The cost of a small slope")
            if let sigmoid, let relu {
                body(String(format: "Sigmoid's derivative peaks at %.2f. Across "
                            + "%d hidden layers that is at best %.3f of the "
                            + "signal reaching the first layer — and only for "
                            + "units sitting exactly at the steepest point. "
                            + "ReLU's peaks at %.2f, so nothing shrinks.",
                            sigmoid.maxSlope, hiddenLayers,
                            pow(sigmoid.maxSlope, Double(hiddenLayers)),
                            relu.maxSlope))
                note(String(format: "Measured on this network: sigmoid needs %@ "
                            + "epochs to reach 95%% accuracy, ReLU needs %@. Same "
                            + "data, same shape, same starting weights — the only "
                            + "difference is the squash.",
                            sigmoid.training.epochsTo95.map(String.init) ?? "—",
                            relu.training.epochsTo95.map(String.init) ?? "—"))
            }

            heading("Choosing one")
            bullet("Hidden layers: ReLU or one of its smooth relatives (GELU, "
                   + "ELU). They do not saturate on the positive side.")
            bullet("Binary output: sigmoid, because a probability is wanted and "
                   + "the range (0, 1) is exactly right.")
            bullet("Multi-class output: softmax, which normalises a whole vector "
                   + "of scores so they sum to one.")
            note("ReLU has its own failure: a unit driven permanently negative "
                 + "has a derivative of exactly zero for every input and never "
                 + "recovers. Leaky ReLU keeps a shallow slope there precisely to "
                 + "avoid that.")
        }
    }

    // MARK: - Backprop

    private var backpropSection: some View {
        let trace = export.backwardTrace
        return VStack(alignment: .leading, spacing: 14) {
            heading("The question backpropagation answers")
            body("For every one of the "
                 + "\(export.architecture.parameterCount) numbers in the network: "
                 + "if this number were nudged slightly, would the loss go up or "
                 + "down, and by how much? That is the gradient, and once it is "
                 + "known every weight can be moved the right way.")

            heading("Why it is not done by brute force")
            body("Each parameter could be nudged one at a time and the loss "
                 + "remeasured, but that costs a full forward pass per parameter. "
                 + "Backpropagation gets every gradient in a single backward "
                 + "sweep, by reusing the chain rule instead of recomputing.")

            heading("The starting point")
            body("A sigmoid output paired with cross-entropy loss collapses to "
                 + "one term — the derivatives cancel:")
            formula("dL/dz = p − y")
            note(String(format: "Here: %.4f − %d = %+.4f. Everything downstream "
                        + "is that one number being distributed.",
                        trace.prediction, trace.label, trace.outputError))

            heading("Then the same three moves, per layer")
            formula("dL/dW = aᵀ · dz        (this layer's weight update)\n"
                    + "dL/db = dz            (its bias update)\n"
                    + "dL/da = dz · Wᵀ       (what to hand back)\n"
                    + "dz    = dL/da ⊙ f′(z) (scaled by the slope below)")
            body("The first says a weight is blamed in proportion to what flowed "
                 + "through it. The last is where the activation function decides "
                 + "how much signal continues.")
            chainWorked

            heading("Why it fades")
            body("Every step back multiplies by a weight matrix and by an "
                 + "activation slope. Both are usually below one, so the signal "
                 + "shrinks with depth — the vanishing gradient. It is why ReLU "
                 + "replaced sigmoid in hidden layers, and why very deep networks "
                 + "need skip connections to give the gradient a shortcut.")
        }
    }

    private var chainWorked: some View {
        let step = export.backwardTrace.steps.first { $0.daPrev != nil }
        guard let step, let da = step.daPrev?.first, let slope = step.primeZ?.first else {
            return AnyView(EmptyView())
        }
        return AnyView(
            note(String(format: "Worked through on one unit: %.4f arriving from "
                        + "the layer above, times a slope of %.4f, leaves %.4f to "
                        + "carry on with.", da, slope, da * slope))
        )
    }

    // MARK: - Training

    private var trainingSection: some View {
        let a = export.architecture
        return VStack(alignment: .leading, spacing: 14) {
            heading("The loop")
            formula("repeat:\n"
                    + "  forward   — predict every point\n"
                    + "  loss      — measure how wrong\n"
                    + "  backward  — get every gradient\n"
                    + "  update    — W ← W − η · dL/dW")
            body(String(format: "Run %d times at a learning rate of %g. Each pass "
                        + "uses every point at once, which is why the loss curve "
                        + "is smooth rather than jagged.", a.epochs, a.learningRate))

            heading("Where the shape comes from")
            body("The first hidden layer can only draw straight cuts — each unit "
                 + "is a logistic regression. The second layer takes weighted "
                 + "combinations of those cuts, and a combination of half-planes "
                 + "can enclose a region. That is how a closed loop appears "
                 + "without anything in the model being told about circles.")
            note("Turn on the hidden-unit maps in the step itself: the first "
                 + "layer's units are visibly straight-edged, and the second "
                 + "layer's are not.")

            heading("What it reached")
            HStack(spacing: 10) {
                metric("accuracy",
                       String(format: "%.1f%%", export.training.finalAccuracy * 100))
                metric("loss", String(format: "%.4f", export.training.finalLoss))
                metric("a straight line",
                       String(format: "%.1f%%", export.dataset.linearBaselineAccuracy * 100))
            }

            heading("What is missing here")
            bullet("No held-out test set — this stage measures fit, not "
                   + "generalisation. The regression stages cover that.")
            bullet("Full-batch descent. Real training uses mini-batches, which "
                   + "are noisier but far cheaper per step.")
            bullet("Plain gradient descent. Adam and friends adapt the step size "
                   + "per parameter and are what is actually used.")
        }
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.subheadline.bold().monospacedDigit())
                .lineLimit(1).minimumScaleFactor(0.6)
            Text(title).font(.caption2).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").font(.subheadline).foregroundStyle(.secondary)
            Text(text).font(.subheadline).foregroundStyle(.secondary)
        }
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
