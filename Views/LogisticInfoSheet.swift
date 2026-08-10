//
//  LogisticInfoSheet.swift
//  VisualML
//
//  The maths behind the logistic stage, worked through with the numbers this
//  app shipped: the model and its odds reading, the cross-entropy loss, the
//  gradient that trains it, and what happens at the decision threshold.
//

import SwiftUI

struct LogisticInfoSheet: View {
    let export: LogisticExport
    var initialTopic: Topic = .setup

    @Environment(\.dismiss) private var dismiss
    @State private var topic: Topic?

    enum Topic: String, CaseIterable, Identifiable {
        case setup = "Setup"
        case model = "Model"
        case loss = "Loss"
        case deciding = "Deciding"
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
                        case .model: modelSection
                        case .loss: lossSection
                        case .deciding: decidingSection
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
        let b = export.linearBaseline
        return VStack(alignment: .leading, spacing: 14) {
            heading("A different kind of target")
            body("Linear regression predicted a number that could be anything. "
                 + "Here the answer is one of two states: \(d.classNames[0].lowercased()) "
                 + "or \(d.classNames[1].lowercased()). Of the \(d.n) students, "
                 + "\(d.positiveCount) passed.")
            body("What we actually want is not the label but the *chance* — a number "
                 + "between 0 and 1 that says how likely this student is to pass.")

            heading("Why a straight line will not do")
            body("Least squares happily fits a line to the 0s and 1s, but nothing "
                 + "in w·x + b keeps the result inside [0, 1]:")
            formula(String(format: "p = %.4f × hours %@ %.4f",
                           b.slope, b.intercept < 0 ? "−" : "+", abs(b.intercept)))
            note(String(format: "On this data it returns values from %.2f to %.2f, "
                        + "and lands outside [0, 1] for %d of the %d students. "
                        + "A −14%% chance is not a chance.",
                        b.minPrediction, b.maxPrediction, b.outOfRangeCount, d.n))

            heading("What is needed instead")
            bullet("Never leave the range 0 to 1, at any input.")
            bullet("Change fastest where the evidence is genuinely mixed.")
            bullet("Flatten out once a case is clear-cut.")
            body("A sigmoid does all three, which is why the next section wraps one "
                 + "around the same linear score.")
        }
    }

    // MARK: - Model

    private var modelSection: some View {
        let fit = export.fit
        return VStack(alignment: .leading, spacing: 14) {
            heading("Score, then squash")
            body("The model keeps the familiar linear part and passes it through "
                 + "one function. The score can be any real number; the squash "
                 + "brings it into 0 to 1.")
            formula("z = w·x + b\np = σ(z) = 1 / (1 + e^−z)")

            heading("The fitted model")
            formula(String(format: "p = σ(%.4f × hours %@ %.4f)",
                           fit.weight, fit.intercept < 0 ? "−" : "+", abs(fit.intercept)))
            body("Worked through for a student who studied 8 hours:")
            formula(workedExample(hours: 8))

            heading("What the weight actually means")
            body("w is not the change in probability — that changes at every point "
                 + "along the curve. Rearranging the model shows what it really is:")
            formula("p / (1 − p) = e^(w·x + b)\n\nlog( p / (1 − p) ) = w·x + b")
            body("The left side is the log-odds. So the model is still perfectly "
                 + "linear, just in log-odds rather than in probability — which is "
                 + "where the name logistic comes from.")
            note(String(format: "e^w = e^%.4f = %.2f, so each extra hour multiplies "
                        + "the odds of passing by %.2f — a bit more than doubling "
                        + "them, wherever you start from.",
                        fit.weight, fit.oddsRatio, fit.oddsRatio))

            heading("Where it tips over")
            body("p = 0.5 happens exactly when the odds are even, which is when "
                 + "z = 0. Solving that for x gives the boundary:")
            formula(String(format: "x = −b / w = %.4f / %.4f = %.2f hours",
                           -fit.intercept, fit.weight, fit.boundary))
        }
    }

    private func workedExample(hours: Double) -> String {
        let fit = export.fit
        let z = fit.weight * hours + fit.intercept
        let p = sigmoid(z)
        return String(format: "z = %.4f × %.0f %@ %.4f = %+.3f\n"
                      + "p = σ(%+.3f) = %.3f\n"
                      + "  → %.0f%% chance of %@",
                      fit.weight, hours, fit.intercept < 0 ? "−" : "+",
                      abs(fit.intercept), z, z, p, p * 100,
                      export.dataset.classNames[1].lowercased())
    }

    // MARK: - Loss

    private var lossSection: some View {
        let fit = export.fit
        return VStack(alignment: .leading, spacing: 14) {
            heading("Scoring a probability")
            body("Squared error measured how far a number was from another number. "
                 + "Here the truth is 0 or 1 and the guess is a probability, so the "
                 + "question becomes: how surprised were we by what happened?")
            formula("J = −(1/n) · Σ [ y·log(p) + (1−y)·log(1−p) ]")
            body("Only one term survives per student. If they passed you pay "
                 + "−log(p); if they failed you pay −log(1−p). Being confident and "
                 + "wrong costs enormously, because log(0) runs to infinity.")
            note(String(format: "The trained model reaches J = %.4f. Guessing 0.5 "
                        + "for everyone would score log(2) = 0.693, so the model is "
                        + "meaningfully better than a coin flip.", fit.logLoss))

            heading("Why not squared error again")
            bullet("Paired with a sigmoid, squared error stops being a single bowl — "
                   + "it develops flat regions and local dips that gradient descent "
                   + "can get stuck in. Cross-entropy stays one bowl.")
            bullet("Squared error treats being 99% wrong as only slightly worse than "
                   + "being 90% wrong. Cross-entropy punishes confident mistakes "
                   + "without limit, which is what you want from a probability.")

            heading("Training it")
            body("Differentiating that loss gives something remarkably tidy — the "
                 + "sigmoid's derivative cancels against the log:")
            formula("∂J/∂θ = (1/n) · Xᵀ (p − y)\n\nθ ← θ − η · ∂J/∂θ")
            note("This is the same shape as linear regression's gradient — error "
                 + "times input — even though the model and the loss are both "
                 + "different. That is not luck: it happens whenever a model is "
                 + "paired with its natural loss.")

            heading("No shortcut this time")
            body("For linear regression we could set the gradient to zero and solve "
                 + "for θ in one step. Try it here and θ stays trapped inside a "
                 + "sigmoid — there is no rearrangement that frees it.")
            note("So gradient descent is not a convenience for logistic regression, "
                 + "it is the only option. There is no closed form to check it "
                 + "against either, which is why this stage has no exact answer "
                 + "drawn alongside.")

            heading("Choosing the step size")
            body("A large rate spoils the fit but cannot blow it up: the error term "
                 + "(p − y) can never exceed 1 in size, so the gradient stays "
                 + "bounded however wrong the parameters get.")
            note(String(format: "That is the opposite of linear regression, where "
                        + "too large a rate sends the loss to infinity. Here η = %g "
                        + "converges; the larger rates in the picker just wander and "
                        + "settle worse.", fit.learningRate))
        }
    }

    // MARK: - Deciding

    private var decidingSection: some View {
        let fit = export.fit
        let c = fit.confusion
        let names = export.dataset.classNames
        return VStack(alignment: .leading, spacing: 14) {
            heading("A probability is not yet an answer")
            body("The model says 71%. Whether that counts as a pass is a separate "
                 + "decision, and it belongs to whoever is using the model.")
            formula(String(format: "predict %@ when p ≥ t\n\nx = ( log( t / (1−t) ) − b ) / w",
                           names[1].lowercased()))
            body(String(format: "With the usual t = 0.5 the log term vanishes and "
                        + "the cut sits at %.2f hours.", fit.boundary))

            heading("Counting the mistakes")
            body("Two ways to be right and two ways to be wrong, on the training set "
                 + "at t = 0.5:")
            confusionGrid(c, names: names)
            body("The diagonal is correct. Off it are the two different mistakes: "
                 + "flagging a student who then failed, and missing one who passed.")

            heading("The two rates worth naming")
            formula("precision = TP / (TP + FP)\nrecall    = TP / (TP + FN)")
            bullet("Precision: of the students we flagged, how many really passed.")
            bullet("Recall: of the students who passed, how many we caught.")
            note("Moving the threshold trades one for the other and never improves "
                 + "both. Dropping it to 0.10 in the previous step catches 99% of "
                 + "passing students, but barely half the flags are right.")

            heading("Is it any good")
            HStack(spacing: 10) {
                metric("accuracy", String(format: "%.1f%%", fit.accuracy * 100))
                metric("held-out", String(format: "%.1f%%", fit.testAccuracy * 100))
                metric("always guess \(names[0].lowercased())",
                       String(format: "%.1f%%", fit.majorityAccuracy * 100))
            }
            body("The held-out number is the one that matters — it is measured on "
                 + "students the model never trained on. Beating the majority-class "
                 + "baseline is the minimum bar for any classifier.")
        }
    }

    private func confusionGrid(_ c: [[Int]], names: [String]) -> some View {
        Grid(horizontalSpacing: 8, verticalSpacing: 6) {
            GridRow {
                Text("").gridColumnAlignment(.leading)
                Text("said \(names[0].lowercased())").font(.caption2).bold()
                Text("said \(names[1].lowercased())").font(.caption2).bold()
            }
            GridRow {
                Text("really \(names[0].lowercased())").font(.caption2).bold()
                confusionCell(c[0][0], label: "TN", correct: true)
                confusionCell(c[0][1], label: "FP", correct: false)
            }
            GridRow {
                Text("really \(names[1].lowercased())").font(.caption2).bold()
                confusionCell(c[1][0], label: "FN", correct: false)
                confusionCell(c[1][1], label: "TP", correct: true)
            }
        }
    }

    private func confusionCell(_ n: Int, label: String, correct: Bool) -> some View {
        VStack(spacing: 1) {
            Text("\(n)").font(.callout.monospacedDigit()).bold()
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background((correct ? Color.green : Color.red).opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func metric(_ title: String, _ value: String) -> some View {
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

    // MARK: - Building blocks

    private func heading(_ text: String) -> some View {
        Text(text).font(.headline).padding(.top, 4)
    }

    private func body(_ text: String) -> some View {
        Text(text).font(.subheadline).foregroundStyle(.secondary)
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").font(.subheadline).foregroundStyle(.secondary)
            Text(text).font(.subheadline).foregroundStyle(.secondary)
        }
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

/// Adds an ⓘ button that opens the maths sheet at a given topic.
struct LogisticInfoButton: ViewModifier {
    let export: LogisticExport
    let topic: LogisticInfoSheet.Topic
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
                LogisticInfoSheet(export: export, initialTopic: topic)
                    .presentationDetents([.large])
            }
    }
}

extension View {
    /// Attaches the maths sheet, opening on the section that matches this step.
    func logisticInfo(_ export: LogisticExport,
                      topic: LogisticInfoSheet.Topic) -> some View {
        modifier(LogisticInfoButton(export: export, topic: topic))
    }
}
