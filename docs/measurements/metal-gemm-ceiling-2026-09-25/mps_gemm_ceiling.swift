// mps_gemm_ceiling.swift — S1 of the Metal prefill GEMM scoping (2026-09-25): what this GPU sustains on the
// 1.5B's prefill GEMM shapes, measured with Apple's own tuned GEMM (MPSMatrixMultiplication, f16 in / f16 out,
// C = A · Wᵀ as goinfer's prefill computes it). It is a CEILING reference, not a peer: f16 weights, no int4
// dequant — goinfer's kernel reads a quarter of the weight bytes and must dequantize them.
//
// Measured the way S0's run 4 says any timing on this GPU must be: under SUSTAINED load. A kernel timed right
// after the GPU has idled runs in a burst (~2x faster for goinfer's gate/up), so the timed loop runs back to
// back after a warm-up, with no idle gap. One extra condition times gate/up after 2 s idle, to show whether the
// burst/sustained split is this GPU's behaviour for any kernel or specific to goinfer's.
//
// Each timed command buffer encodes the shape 28 times (the 1.5B's layer count), cycling 4 distinct weight
// matrices, each larger than the system-level cache, so weights stream from DRAM as they do in production.
//
//   xcrun swiftc -O mps_gemm_ceiling.swift -o mps_gemm_ceiling && ./mps_gemm_ceiling [reps]
import Foundation
import Metal
import MetalPerformanceShaders

let reps = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1]) ?? 5 : 5
let layers = 28, distinctW = 4

guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
    fatalError("no Metal device")
}
print("device: \(dev.name)  reps: \(reps)  layers/cb: \(layers)  distinct weights: \(distinctW)")

struct Shape { let name: String; let n: Int; let k: Int }
let shapes = [Shape(name: "qkv", n: 2048, k: 1536), Shape(name: "o", n: 1536, k: 1536),
              Shape(name: "gate/up", n: 17920, k: 1536), Shape(name: "down", n: 1536, k: 8960)]
let ms = [512, 3904]

func f16Buffer(_ count: Int, seed: UInt32) -> MTLBuffer {
    let b = dev.makeBuffer(length: count * 2, options: .storageModeShared)!
    let p = b.contents().bindMemory(to: Float16.self, capacity: count)
    var s = seed
    for i in 0..<count { s = s &* 1664525 &+ 1013904223; p[i] = Float16(Float(Int32(bitPattern: s >> 16) % 1000) / 8000) }
    return b
}

struct Case { let m: Int; let s: Shape; let a: MPSMatrix; let w: [MPSMatrix]; let c: MPSMatrix; let op: MPSMatrixMultiplication }
var cases: [Case] = []
for m in ms {
    for s in shapes {
        let da = MPSMatrixDescriptor(rows: m, columns: s.k, rowBytes: s.k * 2, dataType: .float16)
        let dw = MPSMatrixDescriptor(rows: s.n, columns: s.k, rowBytes: s.k * 2, dataType: .float16)
        let dc = MPSMatrixDescriptor(rows: m, columns: s.n, rowBytes: s.n * 2, dataType: .float16)
        let a = MPSMatrix(buffer: f16Buffer(m * s.k, seed: 7), descriptor: da)
        let w = (0..<distinctW).map { MPSMatrix(buffer: f16Buffer(s.n * s.k, seed: UInt32(100 + $0)), descriptor: dw) }
        let c = MPSMatrix(buffer: dev.makeBuffer(length: m * s.n * 2, options: .storageModePrivate)!, descriptor: dc)
        let op = MPSMatrixMultiplication(device: dev, transposeLeft: false, transposeRight: true,
                                         resultRows: m, resultColumns: s.n, interiorColumns: s.k, alpha: 1, beta: 0)
        cases.append(Case(m: m, s: s, a: a, w: w, c: c, op: op))
    }
}

// One command buffer: `layers` GEMMs of one shape, cycling the distinct weights. Returns GPU ms.
func run(_ c: Case) -> Double {
    let cb = q.makeCommandBuffer()!
    for l in 0..<layers { c.op.encode(commandBuffer: cb, leftMatrix: c.a, rightMatrix: c.w[l % distinctW], resultMatrix: c.c) }
    cb.commit(); cb.waitUntilCompleted()
    if let e = cb.error { fatalError("command buffer error: \(e)") }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
}
func tflops(_ c: Case, _ msv: Double) -> Double { 2 * Double(c.m * c.s.n * c.s.k * layers) / (msv * 1e-3) / 1e12 }
func median(_ xs: [Double]) -> Double { let s = xs.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }
func spread(_ xs: [Double]) -> Double { (xs.max()! - xs.min()!) / median(xs) }

// Warm-up: >= 3 s of continuous GPU work, so the timed loop starts in the sustained state.
let gu3904 = cases.first { $0.m == 3904 && $0.s.name == "gate/up" }!
let gu512 = cases.first { $0.m == 512 && $0.s.name == "gate/up" }!
let t0 = Date()
while Date().timeIntervalSince(t0) < 3 { _ = run(gu512) }
print(String(format: "warm-up: %.1f s of back-to-back gate/up", Date().timeIntervalSince(t0)))

var sustained = Array(repeating: [Double](), count: cases.count)
var burst: [Double] = []
var afterBurst: [Double] = []
for rep in 0..<reps {
    for (i, c) in cases.enumerated() { sustained[i].append(run(c)) }  // back to back, no gaps
    // Burst check: gate/up at M=512 after 2 s of idle, then immediately again (sustained reference).
    Thread.sleep(forTimeInterval: 2)
    burst.append(run(gu512))
    afterBurst.append(run(gu512))
    for _ in 0..<3 { _ = run(gu3904) }  // back into the sustained state before the next rep
    print(String(format: "rep %d/%d done  (gate/up M=512: after idle %.1f ms, right after %.1f ms)", rep + 1, reps, burst[rep], afterBurst[rep]))
}

print("\n=== MPSMatrixMultiplication f16, sustained, \(layers) GEMMs per command buffer, medians of \(reps) ===")
print("   M   shape      N      K    GPU ms (28x)  spread   TFLOPS   reps (ms)")
for (i, c) in cases.enumerated() {
    let med = median(sustained[i])
    let r = sustained[i].map { String(format: "%.1f", $0) }.joined(separator: " ")
    print(String(format: "%5d  %-8@ %6d %6d   %10.1f   %5.1f%%   %6.2f   %@", c.m, c.s.name as NSString, c.s.n, c.s.k, med, 100 * spread(sustained[i]), tflops(c, med), r))
}
print(String(format: "\nburst check, gate/up M=512: after 2 s idle %.1f ms (%.2f TFLOPS, spread %.1f%%) · immediately after %.1f ms (%.2f TFLOPS, spread %.1f%%)",
             median(burst), tflops(gu512, median(burst)), 100 * spread(burst), median(afterBurst), tflops(gu512, median(afterBurst)), 100 * spread(afterBurst)))
