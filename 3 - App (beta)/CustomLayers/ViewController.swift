import UIKit
import Vision

class ViewController: UIViewController {

  let model = NeuralMcNeuralNet()

  override func viewDidLoad() {
    super.viewDidLoad()

    let image = UIImage(named: "floortje.png")!

    guard let visionModel = try? VNCoreMLModel(for: model.model) else {
      fatalError("whoops")
    }

    let request = VNCoreMLRequest(model: visionModel) { request, error in
      if let observations = request.results as? [VNCoreMLFeatureValueObservation],
         let output = observations.first?.featureValue.multiArrayValue {

        /*
          The output should be the following (or very close to it):
            2.24007443e-02   8.09036642e-02   6.90359762e-03   1.62472681e-03
            6.74153166e-03   7.71020293e-01   1.67265590e-02   3.46490240e-04
            9.12911538e-03   8.42032731e-02
        */
        print(output)
      }
    }

    let handler = VNImageRequestHandler(cgImage: image.cgImage!)
    try? handler.perform([request])
  }
}
