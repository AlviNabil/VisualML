//
//  LogisticFlowView.swift
//  VisualML
//
//  The logistic regression walkthrough. Each stage is one step of the model,
//  in the order the data passes through it, so the pipeline can be followed
//  end to end rather than met all at once.
//

import SwiftUI

struct LogisticFlowView: View {
    @State private var export: LogisticExport?
    @State private var loadError: String?
    @State private var showInfo = false

    var body: some View {
        Group {
            if let export {
                stages(export)
            } else if let loadError {
                ContentUnavailableView("Could not load the model",
                                       systemImage: "exclamationmark.triangle",
                                       description: Text(loadError))
            } else {
                ProgressView("Loading…").frame(maxWidth: .infinity, minHeight: 240)
            }
        }
        .navigationTitle("Logistic Regression")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if let export {
                    Button { showInfo = true } label: { Image(systemName: "info.circle") }
                        .accessibilityLabel("The maths behind this model")
                        .disabled(export.dataset.n == 0)
                }
            }
        }
        .sheet(isPresented: $showInfo) {
            if let export {
                LogisticInfoSheet(export: export).presentationDetents([.large])
            }
        }
        .task {
            do { export = try LogisticExport.load() }
            catch { loadError = error.localizedDescription }
        }
    }

    private func stages(_ export: LogisticExport) -> some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(export.subtitle).font(.subheadline.bold())
                    Text("\(export.dataset.n) students, "
                         + "\(export.dataset.positiveCount) of whom passed. Work "
                         + "through the steps in order — each one picks up where "
                         + "the last left off.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section("Steps") {
                NavigationLink {
                    LogisticDataView(export: export)
                } label: {
                    step("1", "The data",
                         "A yes/no target, and why a line fails", "circle.grid.2x1.fill")
                }
                NavigationLink {
                    LogisticSigmoidView(export: export)
                } label: {
                    step("2", "The transformation",
                         "Score, then squash: z → σ(z)", "function")
                }
                NavigationLink {
                    LogisticTrainingView(export: export)
                } label: {
                    step("3", "Finding the curve",
                         "Gradient descent on the cross-entropy", "arrow.down.right.circle")
                }
                NavigationLink {
                    LogisticDecisionView(export: export)
                } label: {
                    step("4", "The decision",
                         "Where to cut, and what it costs", "checkmark.circle")
                }
            }
        }
    }

    private func step(_ number: String, _ title: String,
                      _ subtitle: String, _ icon: String) -> some View {
        HStack(spacing: 12) {
            Text(number)
                .font(.subheadline.bold())
                .frame(width: 26, height: 26)
                .background(Color.accentColor.opacity(0.15), in: Circle())
                .foregroundStyle(Color.accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body).fontWeight(.semibold)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: icon).font(.footnote).foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}
