//
//  KMeansFlowView.swift
//  VisualML
//
//  The k-means walkthrough. Unlike the regression stages there is no target to
//  predict -- the point is to find structure in unlabeled points -- so the
//  steps run: look at the raw data, choose how many groups to look for, watch
//  the algorithm find them, then see the result on the map.
//

import SwiftUI

struct KMeansFlowView: View {
    @State private var export: ClusterExport?
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
        .navigationTitle("K-Means Clustering")
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
                KMeansInfoSheet(export: export).presentationDetents([.large])
            }
        }
        .task {
            do { export = try ClusterExport.load() }
            catch { loadError = error.localizedDescription }
        }
    }

    private func stages(_ export: ClusterExport) -> some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    Text(export.subtitle).font(.subheadline.bold())
                    Text("\(export.dataset.n) customers, two features, no labels. "
                         + "Work through the steps in order — each one picks up "
                         + "where the last left off.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }

            Section("Steps") {
                NavigationLink {
                    KMeansDataView(export: export)
                } label: {
                    step("1", "The data",
                         "Two features, no labels — nothing to predict yet",
                         "circle.grid.2x1.fill")
                }
                NavigationLink {
                    KMeansElbowView(export: export)
                } label: {
                    step("2", "Choosing k",
                         "The elbow curve: where more clusters stop helping",
                         "chart.line.downtrend.xyaxis")
                }
                NavigationLink {
                    KMeansTrainingView(export: export)
                } label: {
                    step("3", "Finding the clusters",
                         "Assign, then average — watched from a few starting points",
                         "arrow.triangle.2.circlepath")
                }
                NavigationLink {
                    KMeansResultView(export: export)
                } label: {
                    step("4", "The result",
                         "Clusters and centroids on the map",
                         "mappin.and.ellipse")
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
