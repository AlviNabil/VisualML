//
//  BagOfWordsFlowView.swift
//  VisualML
//
//  Stage navigation for the Bag-of-Words text-representation pipeline: jump
//  straight to any stage instead of drilling through them.
//

import SwiftUI

struct BagOfWordsFlowView: View {
    @ObservedObject var viewModel: PipelineViewModel

    var body: some View {
        List {
            Section("Stages") {
                NavigationLink { DatasetView(viewModel: viewModel) } label: {
                    stageRow("1 · Dataset", "\(viewModel.dataPoints.count) labelled documents", "doc.text")
                }
                NavigationLink { BagOfWordsView(viewModel: viewModel) } label: {
                    stageRow("2 · Document-Term Matrix", "Bag-of-Words & TF-IDF heatmap", "square.grid.3x3.fill")
                }
                NavigationLink { LSAView(viewModel: viewModel) } label: {
                    stageRow("3 · LSA projection", "2-D latent space (SVD)", "chart.dots.scatter")
                }
            }
        }
        .navigationTitle("Bag of Words")
    }

    private func stageRow(_ title: String, _ subtitle: String, _ icon: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .frame(width: 30)
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
