//
//  ContentView.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import SwiftUI

struct ContentView: View {
    // Connects the View to the ViewModel
    @StateObject private var viewModel = PipelineViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "text.book.closed")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            
            Text("Matrix Visualizer")
                .font(.title)
                .bold()
            
            Text(viewModel.statusMessage)
                .foregroundColor(.gray)
            
            Button("Load Dataset") {
                viewModel.loadDataset()
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
