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
            2.24579312e-02   6.99496120e-02   7.55519234e-03   1.38940173e-03
            5.51432837e-03   8.00364137e-01   1.42883752e-02   3.57461395e-04
            5.40433871e-03   7.27192238e-02
        */
        print(output)
      }
    }

    let handler = VNImageRequestHandler(cgImage: image.cgImage!)
    try? handler.perform([request])
  }
}
