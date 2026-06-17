//
//  DatasetView.swift
//  VisualML
//
//  The raw dataset stage: lists every document so you can SEE the input before
//  it becomes numbers. (Owned now by HomeView, shown as a navigable stage.)
//

import SwiftUI

struct DatasetView: View {
    @ObservedObject var viewModel: PipelineViewModel

    var body: some View {
        Group {
            if viewModel.dataPoints.isEmpty {
                ProgressView("Loading documents…")
                    .frame(maxWidth: .infinity, minHeight: 220)
            } else {
                list
            }
        }
        .navigationTitle("Dataset")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task { await viewModel.loadDataset(); await viewModel.buildBagOfWords() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
            }
        }
    }

    private var list: some View {
        List {
            statsBar
                .listRowInsets(EdgeInsets())
                .listRowSeparator(.hidden)

            ForEach(viewModel.dataPoints) { point in
                HStack(alignment: .top, spacing: 12) {
                    Text(point.category.capitalized)
                        .font(.caption2).bold()
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(color(for: point.label).opacity(0.15))
                        .foregroundStyle(color(for: point.label))
                        .clipShape(Capsule())
                        .frame(width: 70, alignment: .center)

                    Text(point.text)
                        .font(.subheadline)
                }
                .padding(.vertical, 4)
            }
        }
        .listStyle(.plain)
    }

    private var statsBar: some View {
        HStack {
            Text("\(viewModel.dataPoints.count) documents").font(.headline)
            Spacer()
            ForEach(viewModel.classCounts, id: \.name) { item in
                HStack(spacing: 4) {
                    Circle().fill(color(for: item.label)).frame(width: 8, height: 8)
                    Text("\(item.name.capitalized) \(item.count)")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal).padding(.vertical, 8)
    }

    private func color(for label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }
}
