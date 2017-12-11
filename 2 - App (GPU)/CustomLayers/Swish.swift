import Foundation
import CoreML
import Accelerate
import Metal

@objc(Swish) class Swish: NSObject, MLCustomLayer {
  let swishPipeline: MTLComputePipelineState

  required init(parameters: [String : Any]) throws {
    print(#function, parameters)

    // Create the Metal compute kernels.
    let device = MTLCreateSystemDefaultDevice()!
    let library = device.makeDefaultLibrary()!
    let swishFunction = library.makeFunction(name: "swish")!
    swishPipeline = try! device.makeComputePipelineState(function: swishFunction)

    super.init()
  }

  func setWeightData(_ weights: [Data]) throws {
    print(#function, weights)

    // This layer does not have any learned weights. However, in the conversion
    // script we added some (random) weights anyway, just to see how this works.
    // Here you would copy those weights into a buffer (such as MTLBuffer).
  }

  func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
    print(#function, inputShapes)

    // This layer does not modify the size of the data.
    return inputShapes
  }

  func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
    print(#function, inputs.count, outputs.count)

    // This required method gets called when the model is run on the CPU.

    for i in 0..<inputs.count {
      let input = inputs[i]
      let output = outputs[i]

      // NOTE: In a real app, you might need to handle different datatypes.
      // We only support 32-bit floats for now.
      assert(input.dataType == .float32)
      assert(output.dataType == .float32)
      assert(input.shape == output.shape)

      //print("shape:", input.shape)

      // This is a version of the swish function using a for loop.
      // Useful for debugging but kinda slow.
      /*
      for j in 0..<input.count {
        let x = input[j].floatValue
        let y = x / (1 + exp(-x))
        output[j] = NSNumber(value: y)
      }
      */

      // This is the same code as in the above loop, but using vectorized
      // Accelerate functions. It is much faster.

      let count = input.count
      let inputPointer = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
      let outputPointer = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))

      // output = -input
      vDSP_vneg(inputPointer, 1, outputPointer, 1, vDSP_Length(count))

      // output = exp(-input)
      var countAsInt32 = Int32(count)
      vvexpf(outputPointer, outputPointer, &countAsInt32)

      // output = 1 + exp(-input)
      var one: Float = 1
      vDSP_vsadd(outputPointer, 1, &one, outputPointer, 1, vDSP_Length(count))

      // output = x / (1 + exp(-input))
      vvdivf(outputPointer, inputPointer, outputPointer, &countAsInt32)
    }
  }

  func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {
    print(#function, inputs.count, outputs.count)

    // This method gets called when the model runs on the GPU. It is optional,
    // but recommended that you implemented it -- for the best possible speed!

    print("Pixel format:", inputs[0].pixelFormat.rawValue)
    print("Texture type:", inputs[0].textureType.rawValue)
    print("Texture array length:", inputs[0].arrayLength)

    if let encoder = commandBuffer.makeComputeCommandEncoder() {
      for i in 0..<inputs.count {
        encoder.setTexture(inputs[i], index: 0)
        encoder.setTexture(outputs[i], index: 1)
        encoder.dispatch(pipeline: swishPipeline, texture: inputs[i])
        encoder.endEncoding()
      }
    }
  }
}

extension MTLComputeCommandEncoder {
  public func dispatch(pipeline: MTLComputePipelineState, texture: MTLTexture) {
    let w = pipeline.threadExecutionWidth
    let h = pipeline.maxTotalThreadsPerThreadgroup / w
    let threadGroupSize = MTLSizeMake(w, h, 1)

    let threadGroups = MTLSizeMake(
      (texture.width       + threadGroupSize.width  - 1) / threadGroupSize.width,
      (texture.height      + threadGroupSize.height - 1) / threadGroupSize.height,
      (texture.arrayLength + threadGroupSize.depth  - 1) / threadGroupSize.depth)

    setComputePipelineState(pipeline)
    dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
  }
}
