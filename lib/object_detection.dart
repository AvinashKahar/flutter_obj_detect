import 'dart:developer';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:camera/camera.dart';

class DetectionResult {
  final Rect rect;
  final String label;
  final double confidence;

  DetectionResult(this.rect, this.label, this.confidence);
}

class ObjectDetection {
  //static const String _modelPath = 'assets/custom_ssd_mobilenet_v2.tflite';
  // static const String _modelPath = 'assets/custom_ssd_mobilenet_v2_fpn_lite_320x320.tflite';
  static const String _modelPath = 'assets/ssd_weapon_model_v2.tflite';
  //custom_ssd_mobilenet_v2_fpn_lite_320x320.tflite
  static const String _labelPath = 'assets/labels.txt';

  Interpreter? _interpreter;
  List<String>? _labels;

  ObjectDetection() {
    _loadModel();
    _loadLabels();
    log('Done.');
  }

  Future<void> _loadModel() async {
    log('Loading interpreter options...');
    final interpreterOptions = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      interpreterOptions.addDelegate(XNNPackDelegate());
    }

    // Use Metal Delegate
    if (Platform.isIOS) {
      interpreterOptions.addDelegate(GpuDelegate());
    }

    log('Loading interpreter...');
    _interpreter =
    await Interpreter.fromAsset(_modelPath, options: interpreterOptions);
  }

  Future<void> _loadLabels() async {
    log('Loading labels...');
    final labelsRaw = await rootBundle.loadString(_labelPath);
    _labels = labelsRaw.split('\n');
  }

  Uint8List analyseImage(String imagePath) {
    log('Analysing image...');
    // Reading image bytes from file
    final imageData = File(imagePath).readAsBytesSync();

    // Decoding image
    final image = img.decodeImage(imageData);

    // Resizing image fpr model, [320, 320]
    // If you use SSD_Mobile_Net_V2 FPN Lite 320 x 320 , please change both the width and height to 320
    final imageInput = img.copyResize(
      image!,
      width: 320,
      height: 320,
      // width: 300,
      // height: 300,
    );

    // Creating matrix representation, [320, 320, 3] normalized to [-1, 1]
    final imageMatrix = List.generate(
      imageInput.height,
          (y) => List.generate(
        imageInput.width,
            (x) {
          final pixel = imageInput.getPixel(x, y);
          return [
            ((pixel.r / 127.5) - 1.0),
            ((pixel.g / 127.5) - 1.0),
            ((pixel.b / 127.5) - 1.0),
          ];
        },
      ),
    );

    final output = _runInference(imageMatrix);

    // Process Tensors from the output
    final scoresTensor = output[0].first as List<double>;
    final boxesTensor = output[1].first as List<List<double>>;
    final classesTensor = output[3].first as List<double>;

    log('Processing outputs...');

    // Process bounding boxes
    final List<List<int>> locations = boxesTensor
        .map((box) => box.map((value) => ((value * 300).toInt())).toList())
        .toList();

    log('Processing outputs... locations ${locations.toList()}');

    // Convert class indices to int
    final classes = classesTensor.map((value) => value.toInt()).toList();

    log('Processing outputs... classes ${classes.toList()}');


    // Number of detections
    final numberOfDetections = output[2].first as double;


    log('Processing outputs... numberOfDetections ${numberOfDetections}');

    // Get classifcation with label
    final List<String> classification = [];
    for (int i = 0; i < numberOfDetections; i++) {
      classification.add(_labels![classes[i]]);
      log('Get classifcation with label ${_labels![classes[i]]}');
    }

    log('Outlining objects...');
    for (var i = 0; i < numberOfDetections; i++) {
      if (scoresTensor[i] > 0.4) {
        // Rectangle drawing
        img.drawRect(
          imageInput,
          x1: locations[i][1],
          y1: locations[i][0],
          x2: locations[i][3],
          y2: locations[i][2],
          color: img.ColorRgb8(0, 255, 0),
          thickness: 3,
        );

        // Label drawing
        img.drawString(
          imageInput,
          '${classification[i]} ${scoresTensor[i]}',
          font: img.arial14,
          x: locations[i][1] + 7,
          y: locations[i][0] + 7,
          color: img.ColorRgb8(0, 255, 0),
        );
      }
    }

    log('Done.');
    return img.encodeJpg(imageInput);
  }

  /// Analyses a CameraImage (from the camera package) for object detection.
  /// Returns a JPEG Uint8List with detected boxes and labels drawn.
  //Future<Uint8List> analyseCameraImage(CameraImage cameraImage) async {
  Future<List<DetectionResult>> analyseCameraImage(CameraImage cameraImage) async {
    log('Analysing CameraImage...');
    // Convert YUV420 to RGB using image package
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    // Prepare a buffer for RGB pixels
    final img.Image rgbImage = img.Image(width: width, height: height);

    // YUV420 to RGB conversion (planar)
    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1;

    final Uint8List y = cameraImage.planes[0].bytes;
    final Uint8List u = cameraImage.planes[1].bytes;
    final Uint8List v = cameraImage.planes[2].bytes;

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        final int uvIndex =
            (uvRowStride * (h ~/ 2)) + (uvPixelStride * (w ~/ 2));
        final int yIndex = h * cameraImage.planes[0].bytesPerRow + w;
        final int yp = y[yIndex];
        final int up = u[uvIndex];
        final int vp = v[uvIndex];

        // Convert YUV to RGB
        int r = (yp + (1.370705 * (vp - 128))).round();
        int g = (yp - (0.337633 * (up - 128)) - (0.698001 * (vp - 128))).round();
        int b = (yp + (1.732446 * (up - 128))).round();
        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);
        rgbImage.setPixelRgb(w, h, r, g, b);
      }
    }

    // Resize to 320x320
    final img.Image imageInput = img.copyResize(
      rgbImage,
      width: 320,
      height: 320,
    );

    // Create matrix representation, [320, 320, 3] normalized to [-1, 1]
    final imageMatrix = List.generate(
      imageInput.height,
      (y) => List.generate(
        imageInput.width,
        (x) {
          final pixel = imageInput.getPixel(x, y);
          return [
            ((pixel.r / 127.5) - 1.0),
            ((pixel.g / 127.5) - 1.0),
            ((pixel.b / 127.5) - 1.0),
          ];
        },
      ),
    );

    final output = _runInference(imageMatrix);

    final scoresTensor = output[0].first as List<double>;
    final boxesTensor = output[1].first as List<List<double>>;
    final classesTensor = output[3].first as List<double>;
    final numberOfDetections = output[2].first as double;

    final results = <DetectionResult>[];
    for (int i = 0; i < numberOfDetections; i++) {
      if (scoresTensor[i] > 0.4) {
        final classIndex = classesTensor[i].toInt();
        final label = _labels![classIndex];

        // Bounding boxes are normalized [ymin, xmin, ymax, xmax] in [0,1].
        final box = boxesTensor[i];
        // final rect = Rect.fromLTRB(
        //   box[1] * cameraImage.width,
        //   box[0] * cameraImage.height,
        //   box[3] * cameraImage.width,
        //   box[2] * cameraImage.height,
        // );

        final rect = Rect.fromLTRB(
          box[0], // xmin
          box[1], // ymin
          box[2], // xmax
          box[3], // ymax
        );

        results.add(DetectionResult(rect, label, scoresTensor[i]));
      }
    }
    return results;

    // final output = _runInference(imageMatrix);
    //
    // // Process Tensors from the output
    // final scoresTensor = output[0].first as List<double>;
    // final boxesTensor = output[1].first as List<List<double>>;
    // final classesTensor = output[3].first as List<double>;
    //
    // log('Processing outputs...');
    //
    // // Process bounding boxes
    // final List<List<int>> locations = boxesTensor
    //     .map((box) => box.map((value) => ((value * 300).toInt())).toList())
    //     .toList();
    //
    // // Convert class indices to int
    // final classes = classesTensor.map((value) => value.toInt()).toList();
    //
    // // Number of detections
    // final numberOfDetections = output[2].first as double;
    //
    // // Get classification with label
    // final List<String> classification = [];
    // for (int i = 0; i < numberOfDetections; i++) {
    //   classification.add(_labels![classes[i]]);
    //   log('Get classification with label ${_labels![classes[i]]}');
    // }
    //
    // log('Outlining objects...');
    // for (var i = 0; i < numberOfDetections; i++) {
    //   if (scoresTensor[i] > 0.4) {
    //     // Rectangle drawing
    //     img.drawRect(
    //       imageInput,
    //       x1: locations[i][1],
    //       y1: locations[i][0],
    //       x2: locations[i][3],
    //       y2: locations[i][2],
    //       color: img.ColorRgb8(0, 255, 0),
    //       thickness: 3,
    //     );
    //
    //     // Label drawing
    //     img.drawString(
    //       imageInput,
    //       '${classification[i]} ${scoresTensor[i]}',
    //       font: img.arial14,
    //       x: locations[i][1] + 7,
    //       y: locations[i][0] + 7,
    //       color: img.ColorRgb8(0, 255, 0),
    //     );
    //   }
    // }
    //
    // log('Done.');
    // return img.encodeJpg(imageInput);
  }

  List<List<Object>> _runInference(
      List<List<List<num>>> imageMatrix,
      ) {
    log('Running inference...');

    // Set input tensor [1, 320, 320, 3]
    final input = [imageMatrix];

    // Set output tensor
    // Scores: [1, 10],
    // Locations: [1, 10, 4],
    // Number of detections: [1],
    // Classes: [1, 10],
    final output = {
      0: [List<num>.filled(10, 0)],
      1: [List<List<num>>.filled(10, List<num>.filled(4, 0))],
      2: [0.0],
      3: [List<num>.filled(10, 0)],
    };
    log('Running inference...output ${output}');
    _interpreter!.runForMultipleInputs([input], output);

    log('Running inference...output.values.toList() ${output.values.toList()}');
    return output.values.toList();
  }
}
