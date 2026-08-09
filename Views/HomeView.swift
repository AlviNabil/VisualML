//
//  HomeView.swift
//  VisualML
//
//  The app's landing menu. It OWNS the pipeline view model and loads the dataset
//  once, then offers the ML approaches. Today: Bag of Words (the text pipeline)
//  and Regression (the BoW-based classifier). More are stubbed for later.
//

import SwiftUI

struct HomeView: View {
    // Owned here so every screen shares the same loaded data + caches.
    @StateObject private var viewModel = PipelineViewModel()

    var body: some View {
        NavigationStack {
            List {
                Section("Text representation") {
                    NavigationLink {
                        BagOfWordsFlowView(viewModel: viewModel)
                    } label: {
                        row("Bag of Words", "Documents → matrix → TF-IDF → LSA",
                            "square.grid.3x3.fill", .blue)
                    }
                }

                Section("Regression") {
                    NavigationLink {
                        LinearRegressionView()
                    } label: {
                        row("Linear Regression",
                            "One feature: fit a line, closed form vs gradient descent",
                            "chart.line.uptrend.xyaxis", .green)
                    }
                    NavigationLink {
                        ClassifierView(viewModel: viewModel)
                    } label: {
                        row("Bag-of-Words + Regression",
                            "Logistic / Linear / SVM on the LSA features",
                            "scribble.variable", .orange)
                    }
                }

                Section("More models — coming soon") {
                    comingSoon("Clustering (k-means)", "circle.grid.cross.fill")
                    comingSoon("Decision tree", "arrow.triangle.branch")
                    comingSoon("Neural network", "brain")
                }
            }
            .navigationTitle("VisualML")
        }
        // Load the data + build the matrix once, so any stage is ready instantly.
        .task {
            if viewModel.dataPoints.isEmpty { await viewModel.loadDataset() }
            await viewModel.buildBagOfWords()
        }
    }

    // MARK: - Rows

    private func row(_ title: String, _ subtitle: String, _ icon: String, _ tint: Color) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.title2)
                .frame(width: 34, height: 34)
                .foregroundStyle(.white)
                .background(tint, in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body).fontWeight(.semibold)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private func comingSoon(_ title: String, _ icon: String) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.title2)
                .frame(width: 34, height: 34)
                .foregroundStyle(.secondary)
            Text(title).font(.body).foregroundStyle(.secondary)
            Spacer()
            Text("Soon")
                .font(.caption2).bold()
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(Color.gray.opacity(0.15))
                .clipShape(Capsule())
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    HomeView()
}
