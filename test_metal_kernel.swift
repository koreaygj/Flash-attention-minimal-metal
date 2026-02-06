import Metal
import Foundation

class FlashAttentionBenchmark {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    
    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal not supported")
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        
        // Metal 커널 로드
        let libraryURL = URL(fileURLWithPath: "flash_attention.metallib")
        self.library = try! device.makeLibrary(URL: libraryURL)
    }
    
    func benchmark(seqLen: Int, headDim: Int, numHeads: Int, iterations: Int = 100) {
        // 커널 함수 가져오기
        let basicKernel = library.makeFunction(name: "flash_attn_kernel")!
        let optimizedKernel = library.makeFunction(name: "flash_attn_kernel_optimized")!
        
        // Pipeline 생성
        let basicPipeline = try! device.makeComputePipelineState(function: basicKernel)
        let optimizedPipeline = try! device.makeComputePipelineState(function: optimizedKernel)
        
        // 입력 데이터 준비
        let qkvSize = numHeads * seqLen * headDim
        var qData = [Float](repeating: 0, count: qkvSize)
        var kData = [Float](repeating: 0, count: qkvSize)
        var vData = [Float](repeating: 0, count: qkvSize)
        
        // 랜덤 데이터 생성 (시드 고정으로 재현 가능)
        srand48(42)
        for i in 0..<qkvSize {
            qData[i] = Float(drand48() * 2 - 1)
            kData[i] = Float(drand48() * 2 - 1)
            vData[i] = Float(drand48() * 2 - 1)
        }
        
        // Metal 버퍼 생성
        let qBuffer = device.makeBuffer(bytes: qData, length: qkvSize * MemoryLayout<Float>.size, options: [])!
        let kBuffer = device.makeBuffer(bytes: kData, length: qkvSize * MemoryLayout<Float>.size, options: [])!
        let vBuffer = device.makeBuffer(bytes: vData, length: qkvSize * MemoryLayout<Float>.size, options: [])!
        
        // 출력 버퍼 (각 커널마다 별도)
        let oBufferBasic = device.makeBuffer(length: qkvSize * MemoryLayout<Float>.size, options: [])!
        let oBufferOptimized = device.makeBuffer(length: qkvSize * MemoryLayout<Float>.size, options: [])!
        
        var N = Int32(seqLen)
        var D = Int32(headDim)
        
        // Grid/Thread 설정
        let tileSize = 32
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: numHeads,
            height: (seqLen + tileSize - 1) / tileSize,
            depth: 1
        )
        
        print(String(repeating: "=", count: 70))
        print("Benchmarking Metal Kernels")
        print("Sequence Length: \(seqLen), Head Dim: \(headDim), Heads: \(numHeads)")
        print(String(repeating: "=", count: 70))
        
        // ===== 1. Correctness Check (한 번만 실행) =====
        print("\n[Correctness Check]")
        
        // Basic 커널 실행
        let cmdBufferBasic = commandQueue.makeCommandBuffer()!
        let encoderBasic = cmdBufferBasic.makeComputeCommandEncoder()!
        encoderBasic.setComputePipelineState(basicPipeline)
        encoderBasic.setBuffer(qBuffer, offset: 0, index: 0)
        encoderBasic.setBuffer(kBuffer, offset: 0, index: 1)
        encoderBasic.setBuffer(vBuffer, offset: 0, index: 2)
        encoderBasic.setBuffer(oBufferBasic, offset: 0, index: 3)
        encoderBasic.setBytes(&N, length: MemoryLayout<Int32>.size, index: 4)
        encoderBasic.setBytes(&D, length: MemoryLayout<Int32>.size, index: 5)
        encoderBasic.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoderBasic.endEncoding()
        cmdBufferBasic.commit()
        cmdBufferBasic.waitUntilCompleted()
        
        // Optimized 커널 실행
        let cmdBufferOpt = commandQueue.makeCommandBuffer()!
        let encoderOpt = cmdBufferOpt.makeComputeCommandEncoder()!
        encoderOpt.setComputePipelineState(optimizedPipeline)
        encoderOpt.setBuffer(qBuffer, offset: 0, index: 0)
        encoderOpt.setBuffer(kBuffer, offset: 0, index: 1)
        encoderOpt.setBuffer(vBuffer, offset: 0, index: 2)
        encoderOpt.setBuffer(oBufferOptimized, offset: 0, index: 3)
        encoderOpt.setBytes(&N, length: MemoryLayout<Int32>.size, index: 4)
        encoderOpt.setBytes(&D, length: MemoryLayout<Int32>.size, index: 5)
        encoderOpt.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoderOpt.endEncoding()
        cmdBufferOpt.commit()
        cmdBufferOpt.waitUntilCompleted()
        
        // 결과 비교
        let outputBasic = oBufferBasic.contents().bindMemory(to: Float.self, capacity: qkvSize)
        let outputOptimized = oBufferOptimized.contents().bindMemory(to: Float.self, capacity: qkvSize)
        
        var maxDiff: Float = 0.0
        var meanDiff: Float = 0.0
        var numDiffs = 0
        
        for i in 0..<qkvSize {
            let diff = abs(outputBasic[i] - outputOptimized[i])
            maxDiff = max(maxDiff, diff)
            meanDiff += diff
            
            // 큰 차이가 있는 경우 디버깅 정보 출력 (처음 5개만)
            if diff > 1e-3 && numDiffs < 5 {
                print("  Large diff at index \(i): basic=\(outputBasic[i]), opt=\(outputOptimized[i]), diff=\(diff)")
                numDiffs += 1
            }
        }
        meanDiff /= Float(qkvSize)
        
        print("  Max difference:  \(String(format: "%.2e", maxDiff))")
        print("  Mean difference: \(String(format: "%.2e", meanDiff))")
        
        // 허용 오차 확인
        let tolerance: Float = 1e-4
        if maxDiff < tolerance {
            print("  ✅ Results match! (within tolerance \(tolerance))")
        } else {
            print("  ⚠️  Results differ significantly!")
        }
        
        // ===== 2. Performance Benchmark =====
        print("\n[Performance Benchmark]")
        
        // Basic Kernel
        var basicTimes: [Double] = []
        for _ in 0..<iterations {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            
            encoder.setComputePipelineState(basicPipeline)
            encoder.setBuffer(qBuffer, offset: 0, index: 0)
            encoder.setBuffer(kBuffer, offset: 0, index: 1)
            encoder.setBuffer(vBuffer, offset: 0, index: 2)
            encoder.setBuffer(oBufferBasic, offset: 0, index: 3)
            encoder.setBytes(&N, length: MemoryLayout<Int32>.size, index: 4)
            encoder.setBytes(&D, length: MemoryLayout<Int32>.size, index: 5)
            
            let start = CFAbsoluteTimeGetCurrent()
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            basicTimes.append(elapsed)
        }
        
        let avgBasic = basicTimes.reduce(0, +) / Double(iterations)
        let minBasic = basicTimes.min()!
        let maxBasic = basicTimes.max()!
        
        print("  Basic Kernel:")
        print("    Avg: \(String(format: "%.4f", avgBasic)) ms")
        print("    Min: \(String(format: "%.4f", minBasic)) ms")
        print("    Max: \(String(format: "%.4f", maxBasic)) ms")
        
        // Optimized Kernel
        var optimizedTimes: [Double] = []
        for _ in 0..<iterations {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            
            encoder.setComputePipelineState(optimizedPipeline)
            encoder.setBuffer(qBuffer, offset: 0, index: 0)
            encoder.setBuffer(kBuffer, offset: 0, index: 1)
            encoder.setBuffer(vBuffer, offset: 0, index: 2)
            encoder.setBuffer(oBufferOptimized, offset: 0, index: 3)
            encoder.setBytes(&N, length: MemoryLayout<Int32>.size, index: 4)
            encoder.setBytes(&D, length: MemoryLayout<Int32>.size, index: 5)
            
            let start = CFAbsoluteTimeGetCurrent()
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            optimizedTimes.append(elapsed)
        }
        
        let avgOptimized = optimizedTimes.reduce(0, +) / Double(iterations)
        let minOptimized = optimizedTimes.min()!
        let maxOptimized = optimizedTimes.max()!
        
        print("  Optimized Kernel:")
        print("    Avg: \(String(format: "%.4f", avgOptimized)) ms")
        print("    Min: \(String(format: "%.4f", minOptimized)) ms")
        print("    Max: \(String(format: "%.4f", maxOptimized)) ms")
        
        let speedup = avgBasic / avgOptimized
        print("\n  Speedup: \(String(format: "%.2f", speedup))x")
        
        // 메모리 처리량 계산
        let bytesPerRun = Float(qkvSize * MemoryLayout<Float>.size * 4) // Q, K, V, O
        let gbPerSec = Double(bytesPerRun) / (avgOptimized / 1000.0) / 1e9
        print("  Memory Bandwidth: \(String(format: "%.2f", gbPerSec)) GB/s")
        
        print()
    }
}

// 실행
print("Flash Attention Metal Kernel Benchmark\n")

let benchmark = FlashAttentionBenchmark()

// 다양한 시퀀스 길이 테스트
for seqLen in [128, 256, 512, 1024, 2048] {
    benchmark.benchmark(seqLen: seqLen, headDim: 64, numHeads: 12, iterations: 50)
}

print(String(repeating: "=", count: 70))
print("Benchmark Complete!")
print(String(repeating: "=", count: 70))
