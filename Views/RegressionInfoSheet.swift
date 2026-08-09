//
//  RegressionInfoSheet.swift
//  VisualML
//
//  The maths behind the regression stages, worked through with the numbers this
//  app actually shipped: the closed-form solution, gradient descent, and what
//  changes when a second feature is added.
//

import SwiftUI

struct RegressionInfoSheet: View {
    let export: RegressionExport
    @Environment(\.dismiss) private var dismiss
    @State private var topic: Topic = .setup

    enum Topic: String, CaseIterable, Identifiable {
        case setup = "Setup"
        case closedForm = "Closed form"
        case gradient = "Gradient descent"
        case multiple = "Multiple"
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker("Topic", selection: $topic) {
                    ForEach(Topic.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding()

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        switch topic {
                        case .setup: setupSection
                        case .closedForm: closedFormSection
                        case .gradient: gradientSection
                        case .multiple: multipleSection
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
        let features = d.featureNames.count
        return VStack(alignment: .leading, spacing: 14) {
            heading("What we are fitting")
            body("Each of the \(d.n) students gives one row: \(features == 1 ? "one input" : "\(features) inputs") "
                 + "and the value we want to predict, \(d.targetLabel.lowercased()).")

            formula(features == 1
                    ? "ŷ = w·x + b"
                    : "ŷ = w₁·x₁ + w₂·x₂ + b")

            heading("Folding b into the parameters")
            body("Rather than carry the intercept separately, every row gets an "
                 + "extra column of 1s. The model is then a single dot product, "
                 + "and the same formulas work for any number of features.")
            formula("X = [\(d.featureNames.map { _ in "x" }.joined(separator: ", "))\(d.featureNames.isEmpty ? "" : ", ")1]"
                    + "      θ = [\(d.featureNames.map { _ in "w" }.joined(separator: ", "))\(d.featureNames.isEmpty ? "" : ", ")b]"
                    + "\n\nŷ = X · θ")
            body("With \(d.n) rows and \(features + 1) parameters, X is \(d.n)×\(features + 1) and θ has \(features + 1) entries.")

            heading("What counts as a good fit")
            body("The mean squared error: average the squared gap between each "
                 + "prediction and the truth. Squaring makes every miss positive "
                 + "and punishes one large error more than several small ones.")
            formula("J(θ) = (1/n) · Σ (ŷᵢ − yᵢ)²")
            note("Always predicting the mean (\(fmt(export.baseline.prediction))) gives "
                 + "J = \(fmt(export.baseline.mse)). The fitted model reaches "
                 + "\(fmt(export.closedForm.mse)) — that gap is what the model learned.")
        }
    }

    // MARK: - Closed form

    private var closedFormSection: some View {
        let n = export.dataset.trainCount
        let q = export.lossQuadratic
        let theta = export.closedForm.theta
        return VStack(alignment: .leading, spacing: 14) {
            heading("Solving it exactly")
            body("J is a smooth bowl in θ with exactly one lowest point. At that "
                 + "point the surface is flat in every direction, so every partial "
                 + "derivative is zero. Setting them to zero gives one equation per "
                 + "parameter — the normal equations.")
            formula("∇J = 0   ⟹   XᵀX · θ = Xᵀy")

            heading("The numbers here")
            body("Both sides are built from the \(n) training rows. Shown divided "
                 + "by n, which does not change the solution:")
            matrixBlock("XᵀX / n", q.gram)
            vectorBlock("Xᵀy / n", q.xty)

            heading("The solution")
            body("Solving that system gives θ directly — no iteration, no learning "
                 + "rate, no stopping rule.")
            formula(thetaLine(theta))
            interpretation

            heading("Why not always use it")
            body("Solving costs roughly d³ work for d features, so it stops being "
                 + "practical when d is large, and it needs XᵀX to be invertible. "
                 + "It also only exists because the squared-error loss is quadratic: "
                 + "logistic regression has no closed form at all.")
        }
    }

    private var interpretation: some View {
        let names = export.dataset.axisLabels
        let weights = export.closedForm.weights
        return VStack(alignment: .leading, spacing: 6) {
            ForEach(names.indices, id: \.self) { i in
                note("Each extra \(singular(names[i])) is worth "
                     + "\(fmt(weights[i])) \(export.dataset.targetLabel.lowercased()) points.")
            }
            note("With everything at zero the model starts from "
                 + "\(fmt(export.closedForm.intercept)).")
        }
    }

    // MARK: - Gradient descent

    private var gradientSection: some View {
        let theta = export.closedForm.theta
        let zero = [Double](repeating: 0, count: theta.count)
        let g0 = export.lossQuadratic.gradient(zero)
        return VStack(alignment: .leading, spacing: 14) {
            heading("Walking downhill instead")
            body("Start anywhere and repeatedly step against the slope. The "
                 + "gradient points in the direction the loss climbs fastest, so "
                 + "moving the other way lowers it.")
            formula("∂J/∂θ = (2/n) · Xᵀ(Xθ − y)\n\nθ ← θ − η · ∂J/∂θ")

            heading("The first step")
            body("Training starts at θ = 0, a flat line at zero. The gradient there is:")
            vectorBlock("∂J/∂θ at θ = 0", g0)
            body("Every entry is negative, so the first step raises all the "
                 + "parameters — the line lifts off the floor towards the data.")

            heading("Choosing η")
            body("The step size decides everything. Too small and it crawls; too "
                 + "large and each step overshoots by more than it gained, so the "
                 + "loss grows without bound.")
            note("The rates in the picker are shown as multiples of the largest "
                 + "stable step. Anything at or above 1.00× diverges, which is why "
                 + "the last option blows up.")

            heading("Reading the parabola")
            body("Holding every parameter but one fixed and sweeping the remaining "
                 + "one traces a parabola — a slice through the bowl. The tangent's "
                 + "steepness at your current position is exactly the gradient for "
                 + "that parameter, and the step length is η times that slope.")
            note("Near the bottom the tangent flattens, so the steps shrink on "
                 + "their own and the fit settles rather than bouncing.")

            heading("Where it ends up")
            body("Given a stable η and enough iterations, it lands on the same "
                 + "answer the closed form produced:")
            formula(thetaLine(theta))
        }
    }

    // MARK: - Multiple regression

    private var multipleSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            heading("Adding a second feature")
            body("Nothing about the maths changes. X gains a column, θ gains an "
                 + "entry, and the same two solvers run untouched — that is the "
                 + "payoff of folding b into θ.")
            formula("one feature   ŷ = w·x + b        θ = [w, b]\n"
                    + "two features  ŷ = w₁x₁ + w₂x₂ + b   θ = [w₁, w₂, b]")

            heading("From a line to a plane")
            body("With one input the fit is a line through a flat scatter. With two "
                 + "it is a plane through a cloud: the height above the floor is the "
                 + "prediction, and the two floor axes are the features.")
            note("A third feature would make it a 3-D hyperplane inside 4-D space — "
                 + "the algebra still works, but the picture runs out.")

            heading("Reading the weights")
            body("Each weight is the effect of its own feature with the other held "
                 + "fixed: the change in prediction per unit, all else equal.")
            if export.dataset.featureNames.count > 1 {
                interpretation
            } else {
                note("This screen shows the single-feature model. Open Multiple "
                     + "Linear Regression from the home menu to see the plane.")
            }

            heading("Does it help?")
            body("Adding a feature can never raise the training error, so the "
                 + "honest question is whether the held-out score improves too.")
            metricRow("R² train", fmt(export.closedForm.r2))
            metricRow("R² test", fmt(export.closedForm.testR2))
            note(export.dataset.featureNames.count > 1
                 ? "Hours alone reaches about 0.50. Adding sleep lifts it to "
                   + "\(fmt(export.closedForm.r2)) on the same target — and the test "
                   + "score moves with it, so the gain is real rather than memorised."
                 : "One feature explains about half the variation. The rest is "
                   + "noise plus whatever drivers this model cannot see.")

            heading("The cost")
            body("More features make the bowl more lopsided. Here the steepest "
                 + "direction is thousands of times steeper than the shallowest, so "
                 + "gradient descent needed 50,000 iterations to settle where the "
                 + "single-feature fit needed 2,000. The closed form is unaffected.")
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

    private func metricRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospacedDigit().bold())
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(Color.gray.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private func matrixBlock(_ title: String, _ rows: [[Double]]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                ForEach(rows.indices, id: \.self) { r in
                    HStack(spacing: 10) {
                        ForEach(rows[r].indices, id: \.self) { c in
                            Text(fmt(rows[r][c]))
                                .font(.caption2.monospacedDigit())
                                .frame(minWidth: 62, alignment: .trailing)
                        }
                    }
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func vectorBlock(_ title: String, _ values: [Double]) -> some View {
        matrixBlock(title, [values])
    }

    // MARK: - Formatting

    private func fmt(_ value: Double) -> String {
        abs(value) >= 10000 ? String(format: "%.1e", value) : String(format: "%.3f", value)
    }

    private func thetaLine(_ theta: [Double]) -> String {
        let names = export.dataset.axisLabels
        let terms = names.indices
            .map { String(format: "%.3f × %@", theta[$0], names[$0].lowercased()) }
            .joined(separator: "  +  ")
        return "\(export.dataset.targetLabel.lowercased()) = \(terms)  +  "
            + String(format: "%.3f", theta.last ?? 0)
    }

    /// "Hours studied" -> "hour studied", so a sentence can say "each extra ...".
    private func singular(_ label: String) -> String {
        var parts = label.lowercased().split(separator: " ").map(String.init)
        if let first = parts.first, first.hasSuffix("s") {
            parts[0] = String(first.dropLast())
        }
        return parts.joined(separator: " ")
    }
}
