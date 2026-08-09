//
//  RegressionDataTableView.swift
//  VisualML
//
//  The dataset behind a regression stage, row by row: the feature values, the
//  observed target, what the fitted model predicts, and the residual between
//  them. Works for any number of features.
//

import SwiftUI

struct RegressionDataTableView: View {
    let export: RegressionExport
    /// The parameters used for the predicted / residual columns.
    let solution: RegressionSolution

    @State private var scope: Scope = .all
    @State private var sort: SortField = .index

    enum Scope: String, CaseIterable, Identifiable {
        case all = "All", train = "Train", test = "Test"
        var id: String { rawValue }
    }

    enum SortField: String, CaseIterable, Identifiable {
        case index = "Row", feature = "Feature", target = "Actual", error = "|Error|"
        var id: String { rawValue }
    }

    /// One prepared row of the table.
    private struct Row: Identifiable {
        let id: Int
        let x: [Double]
        let actual: Double
        let predicted: Double
        var residual: Double { actual - predicted }
        let isTest: Bool
    }

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider()
            headerRow
            Divider()
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(rows) { row in
                        dataRow(row)
                        Divider().opacity(0.4)
                    }
                }
            }
            Divider()
            summary
        }
        .navigationTitle("Dataset")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Rows

    private var rows: [Row] {
        let all = export.dataset.points.enumerated().map { index, point in
            Row(id: index, x: point.x, actual: point.y,
                predicted: solution.predict(point.x), isTest: point.test)
        }
        let scoped: [Row]
        switch scope {
        case .all: scoped = all
        case .train: scoped = all.filter { !$0.isTest }
        case .test: scoped = all.filter { $0.isTest }
        }
        switch sort {
        case .index: return scoped
        case .feature: return scoped.sorted { $0.x[0] < $1.x[0] }
        case .target: return scoped.sorted { $0.actual < $1.actual }
        case .error: return scoped.sorted { abs($0.residual) > abs($1.residual) }
        }
    }

    // MARK: - Pieces

    private var controls: some View {
        VStack(spacing: 8) {
            Picker("Scope", selection: $scope) {
                ForEach(Scope.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            Picker("Sort", selection: $sort) {
                ForEach(SortField.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
    }

    private var headerRow: some View {
        HStack(spacing: 0) {
            cell("#", width: 34, bold: true)
            ForEach(export.dataset.axisLabels.indices, id: \.self) { i in
                cell(shortLabel(export.dataset.axisLabels[i]), bold: true)
            }
            cell("actual", bold: true)
            cell("pred", bold: true)
            cell("resid", bold: true)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.gray.opacity(0.12))
    }

    private func dataRow(_ row: Row) -> some View {
        HStack(spacing: 0) {
            HStack(spacing: 3) {
                Text("\(row.id)").font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                Circle().fill(row.isTest ? Color.green : Color.blue)
                    .frame(width: 5, height: 5)
            }
            .frame(width: 34, alignment: .leading)

            ForEach(row.x.indices, id: \.self) { i in
                cell(String(format: "%.2f", row.x[i]))
            }
            cell(String(format: "%.1f", row.actual))
            cell(String(format: "%.1f", row.predicted))
            Text(String(format: "%+.1f", row.residual))
                .font(.caption.monospacedDigit())
                .foregroundStyle(residualColor(row.residual))
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
    }

    private var summary: some View {
        let shown = rows
        let meanAbs = shown.isEmpty ? 0
            : shown.reduce(0) { $0 + abs($1.residual) } / Double(shown.count)
        return HStack {
            Text("\(shown.count) rows").font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(String(format: "mean |residual| %.2f", meanAbs))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.08))
    }

    // MARK: - Helpers

    private func cell(_ text: String, width: CGFloat? = nil, bold: Bool = false) -> some View {
        Text(text)
            .font(bold ? .caption2.bold() : .caption.monospacedDigit())
            .foregroundStyle(bold ? .secondary : .primary)
            .frame(width: width, alignment: width == nil ? .trailing : .leading)
            .frame(maxWidth: width == nil ? .infinity : nil,
                   alignment: width == nil ? .trailing : .leading)
    }

    /// Trims a long axis label down to something that fits a column head.
    private func shortLabel(_ label: String) -> String {
        label.split(separator: " ").first.map(String.init)?.lowercased() ?? label
    }

    private func residualColor(_ value: Double) -> Color {
        abs(value) < 5 ? .secondary : (value > 0 ? .orange : .purple)
    }
}
