import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DetectionResult {
  final Rect rect; // normalized coords: left, top, right, bottom (0..1)
  final String label;
  final double confidence;

  DetectionResult(this.rect, this.label, this.confidence);
}

class ObjectDetection {
  // Default asset model path (keeps backward compatibility)
  //static const String _modelAssetPath = 'assets/ssd_weapon_model_v2.tflite';
  //static const String _labelPath = 'assets/labels.txt';

  Interpreter? _interpreter;
  List<String>? _labels;

  ObjectDetection() {
    // Start async loading (constructor cannot await). Call ensureLoaded() before inference.
    _loadModelFromAsset(); // non-blocking call
    _loadLabels();
    log('ObjectDetection constructed (loading started).');
  }

  List<String>? get labels => _labels;

  /// Ensure interpreter and labels are ready. Call before starting the camera stream.
  Future<void> ensureLoaded() async {
    if (_interpreter == null) await _loadModelFromAsset();
    if (_labels == null) await _loadLabels();
  }

  /// Loads model that is packaged as an asset.
  Future<void> _loadModelFromAsset() async {
    if (_interpreter != null) return;
    log('Loading interpreter options...');
    final options = InterpreterOptions();

    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }
    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    log('Loading interpreter from asset...');
    //_interpreter = await Interpreter.fromAsset(_modelAssetPath, options: options);
    _interpreter = await Interpreter.fromFile(File("sdcard/Download/ssd_weapon_model_v2.tflite"));
    log('Interpreter loaded from asset.');
  }

  /// If you'd rather load model from external storage path, call this with the file path.
  Future<void> loadModelFromFile(String path) async {
    if (_interpreter != null) {
      // If already loaded, ignore or close & reload
      return;
    }
    final file = File(path);
    if (!file.existsSync()) {
      throw Exception('Model file not found at $path');
    }
    log('Loading interpreter from file: $path');
    final options = InterpreterOptions();
    if (Platform.isAndroid) options.addDelegate(XNNPackDelegate());
    if (Platform.isIOS) options.addDelegate(GpuDelegate());
    _interpreter = await Interpreter.fromFile(file, options: options);
    log('Interpreter loaded from file.');
  }

  Future<void> _loadLabels() async {
    if (_labels != null) return;
    log('Loading labels...');
    final raw = await File("sdcard/Download/labels.txt").readAsString();
    //final raw = await rootBundle.loadString(_labelPath);
    _labels = raw.split('\n').where((s) => s.trim().isNotEmpty).toList();
    log('Labels loaded (${_labels?.length}).');
  }

  /// Public wrapper: caller has preprocessed matrix [320][320][3] normalized to [-1,1]
  List<List<Object>> runInferenceFromImageMatrix(List<List<List<num>>> imageMatrix) {
    return _runInference(imageMatrix);
  }

  /// STILL IMAGE pipeline - unchanged (synchronous-ish)
  Uint8List analyseImageFile(String imagePath) {
    final bytes = File(imagePath).readAsBytesSync();
    final image = img.decodeImage(bytes)!;
    final resized = img.copyResize(image, width: 320, height: 320);

    final imageMatrix = List.generate(
      resized.height,
          (y) => List.generate(
        resized.width,
            (x) {
          final p = resized.getPixel(x, y);
          return [
            ((p.r / 127.5) - 1.0),
            ((p.g / 127.5) - 1.0),
            ((p.b / 127.5) - 1.0),
          ];
        },
      ),
    );

    final output = _runInference(imageMatrix);

    // You can reuse existing code to draw boxes on the image if needed.
    final scores = output[0].first as List<double>;
    final boxes = output[1].first as List<List<double>>;
    final classes = output[3].first as List<double>;
    final numDetections = (output[2].first as num).toInt();

    final locations = boxes
        .map((b) => b.map((v) => (v * 320).toInt()).toList())
        .toList(); // not used here further

    // draw boxes on resized as needed...
    return img.encodeJpg(resized);
  }

  /// Internal inference runner (reuses the interpreter).
  List<List<Object>> _runInference(List<List<List<num>>> imageMatrix) {
    if (_interpreter == null) throw Exception('Interpreter not initialized');

    final input = [imageMatrix];

    final output = {
      0: [List<double>.filled(10, 0.0)], // scores
      1: [List<List<double>>.filled(10, List<double>.filled(4, 0.0))], // boxes
      2: [0.0], // num detections
      3: [List<double>.filled(10, 0.0)], // classes
    };

    _interpreter!.runForMultipleInputs([input], output);

    // Convert typed lists to runtime types expected by caller
    return output.values.toList();
  }
}

/// Top-level function — required by `compute()` (must be top-level or static).
/// Converts CameraImage plane bytes to a normalized image matrix [320][320][3].
List<List<List<num>>> runConvertCameraImageToMatrix(Map<String, dynamic> params) {
  final int width = params['width'] as int;
  final int height = params['height'] as int;
  final List<dynamic> planesDynamic = params['planes'] as List<dynamic>;

  final Uint8List y = planesDynamic[0] as Uint8List;
  final Uint8List u = planesDynamic[1] as Uint8List;
  final Uint8List v = planesDynamic[2] as Uint8List;

  final List<dynamic> bytesPerRowList = params['bytesPerRow'] as List<dynamic>;
  final List<dynamic> bytesPerPixelList = params['bytesPerPixel'] as List<dynamic>;

  final int yRowStride = bytesPerRowList[0] as int;
  final int uvRowStride = bytesPerRowList[1] as int;
  final int uvPixelStride = bytesPerPixelList[1] as int;

  final img.Image rgbImage = img.Image(width: width, height: height);

  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      final int uvIndex = (uvRowStride * (h ~/ 2)) + (uvPixelStride * (w ~/ 2));
      final int yIndex = h * yRowStride + w;

      final int yp = y[yIndex];
      final int up = u[uvIndex];
      final int vp = v[uvIndex];

      int r = (yp + (1.370705 * (vp - 128))).round();
      int g = (yp - (0.337633 * (up - 128)) - (0.698001 * (vp - 128))).round();
      int b = (yp + (1.732446 * (up - 128))).round();

      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      rgbImage.setPixelRgb(w, h, r, g, b);
    }
  }

  final img.Image imageInput = img.copyResize(rgbImage, width: 320, height: 320);

  final imageMatrix = List.generate(
    imageInput.height,
        (yCoord) => List.generate(
      imageInput.width,
          (xCoord) {
        final pixel = imageInput.getPixel(xCoord, yCoord);
        return [((pixel.r / 127.5) - 1.0), ((pixel.g / 127.5) - 1.0), ((pixel.b / 127.5) - 1.0)];
      },
    ),
  );

  return imageMatrix;
}