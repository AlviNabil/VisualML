//
//  DataPoint.swift
//  VisualML
//
//  Created by Alvi Ahmmed Nabil on 6/12/26.
//

import Foundation

struct DataPoint : Identifiable
{
    let id = UUID()
    let text: String
    let label: Int
}
