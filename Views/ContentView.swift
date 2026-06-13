//
//  ContentView.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import SwiftUI

/// The first screen. Right now it loads the dataset and lists every document
/// so you can SEE the raw input before we start turning it into numbers.
struct ContentView: View {
    // @StateObject creates and OWNS the view model for this screen's lifetime.
    // When the VM's @Published values change, SwiftUI re-runs `body` for us.
    @StateObject private var viewModel = PipelineViewModel()

    var body: some View {
        // NavigationStack gives us a title bar and a place for toolbar buttons.
        NavigationStack {
            // `Group` just lets us pick ONE of two layouts without extra nesting.
            Group {
                if viewModel.dataPoints.isEmpty {
                    emptyState          // nothing loaded yet (or an error)
                } else {
                    loadedView          // documents are ready to show
                }
            }
            .navigationTitle("VisualML")
            .toolbar {
                // A reload button in the top-right of the navigation bar.
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.loadDataset()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
        }
        // `.onAppear` runs when the screen first shows. We auto-load once so the
        // app is useful immediately instead of starting on an empty screen.
        .onAppear {
            if viewModel.dataPoints.isEmpty {
                viewModel.loadDataset()
            }
        }
    }

    // MARK: - Subviews
    // Breaking `body` into small computed properties keeps each piece readable.

    /// Shown before any data is loaded, or if loading failed.
    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "tray.and.arrow.down")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text(viewModel.statusMessage)
                .foregroundStyle(.secondary)
            Button("Load Dataset") { viewModel.loadDataset() }
                .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    /// Shown once documents are loaded: a stats bar on top, scrollable list below.
    private var loadedView: some View {
        VStack(spacing: 0) {
            statsBar
            Divider()
            // `List` is iOS's efficient scrolling table. Because DataPoint is
            // Identifiable, we can hand the array straight to List.
            List(viewModel.dataPoints) { point in
                HStack(alignment: .top, spacing: 12) {
                    // A small colored "chip" showing the class.
                    Text(point.category.capitalized)
                        .font(.caption2).bold()
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(color(for: point.label).opacity(0.15))
                        .foregroundStyle(color(for: point.label))
                        .clipShape(Capsule())

                    Text(point.text)
                        .font(.subheadline)
                }
                .padding(.vertical, 4)
            }
            .listStyle(.plain)
        }
    }

    /// A header summarizing how many documents and how many per class.
    private var statsBar: some View {
        HStack {
            Text("\(viewModel.dataPoints.count) documents")
                .font(.headline)
            Spacer()
            // One colored dot + count per class (the legend).
            ForEach(viewModel.classCounts, id: \.name) { item in
                HStack(spacing: 4) {
                    Circle()
                        .fill(color(for: item.label))
                        .frame(width: 8, height: 8)
                    Text("\(item.name.capitalized) \(item.count)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    // MARK: - Helpers

    /// Map a numeric class label to a stable color (class 0 = blue, 1 = orange, ...).
    private func color(for label: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .purple, .pink]
        return palette[label % palette.count]
    }
}

#Preview {
    ContentView()
}
