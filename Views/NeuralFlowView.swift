//
//  NeuralFlowView.swift
//  VisualML
//
//  The neural network walkthrough. The steps follow the order the numbers
//  actually move: look at a problem nothing so far can solve, meet the shape
//  of the model, push one point forward through it, see what the activation
//  does, then send the error back and watch the boundary form.
//

import SwiftUI

struct NeuralFlowView: View {
    @State private var export: NeuralExport?
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
        .navigationTitle("Neural Network")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if export != nil {
                    Button { showInfo = true } label: { Image(systemName: "info.circle") }
                        .accessibilityLabel("The maths behind this model")
                }
            }
        }
        .sheet(isPresented: $showInfo) {
            if let export {
                NeuralInfoSheet(export: export).presentationDetents([.large])
            }
        }
        .task {
            do { export = try NeuralExport.load() }
            catch { loadError = error.localizedDescription }
        }
    }

    private func stages(_ export: NeuralExport) -> some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(export.subtitle).font(.subheadline.bold())
                    Text("A \(export.architecture.layerSizes.map(String.init).joined(separator: "→")) "
                         + "network with \(export.architecture.parameterCount) parameters, "
                         + "small enough that every number in it can be shown. "
                         + "Work through the steps in order.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section("Steps") {
                NavigationLink {
                    NeuralDataView(export: export)
                } label: {
                    step("1", "The data",
                         "A boundary that curves — where a line fails",
                         "circle.circle")
                }
                NavigationLink {
                    NeuralArchitectureView(export: export)
                } label: {
                    step("2", "The architecture",
                         "Layers, weights, and what the model stores",
                         "square.stack.3d.up")
                }
                NavigationLink {
                    NeuralForwardView(export: export)
                } label: {
                    step("3", "The forward pass",
                         "One point through every layer, with the arithmetic",
                         "arrow.right")
                }
                NavigationLink {
                    NeuralActivationsView(export: export)
                } label: {
                    step("4", "Activations",
                         "Six squashes, their slopes, and why it matters",
                         "function")
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
