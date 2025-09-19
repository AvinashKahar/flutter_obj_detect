// lib/object_detection.dart
import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Small detection result used by painter & UI
class DetectionResult {
  final Rect rect; // normalized coords left, top, right, bottom
  final String label;
  final double confidence;
  DetectionResult(this.rect, this.label, this.confidence);
}

class ObjectDetection {
  Interpreter? _interpreter;
  List<String>? _labels;

  ObjectDetection();

  /// Load model from external storage (sdcard)
  Future<void> loadModelFromFile(String path) async {
    final f = File(path);
    if (!f.existsSync()) {
      throw Exception('Model file not found at $path');
    }
    final opts = InterpreterOptions();
    if (Platform.isAndroid) opts.addDelegate(XNNPackDelegate());
    if (Platform.isIOS) opts.addDelegate(GpuDelegate());
    _interpreter = await Interpreter.fromFile(f, options: opts);
    log('Interpreter loaded from: $path');
  }

  /// Load labels from external file
  Future<void> loadLabelsFromFile(String path) async {
    final f = File(path);
    if (!f.existsSync()) {
      log('Labels file not found at $path');
      return;
    }
    final raw = await f.readAsString();
    _labels = raw.split('\n').where((s) => s.trim().isNotEmpty).toList();
    log('Labels loaded (${_labels?.length}) from $path');
  }

  /// Fallback: ensure interpreter (load from asset if no file loaded)
  Future<void> ensureInterpreterInitialized() async {
    if (_interpreter == null) {
      final opts = InterpreterOptions();
      if (Platform.isAndroid) opts.addDelegate(XNNPackDelegate());
      if (Platform.isIOS) opts.addDelegate(GpuDelegate());
      _interpreter = await Interpreter.fromAsset('assets/ssd_weapon_model_v2.tflite', options: opts);
      log('Interpreter loaded from asset fallback.');
    }
    if (_labels == null) {
      try {
        final raw = await rootBundle.loadString('assets/labels.txt');
        _labels = raw.split('\n').where((s) => s.trim().isNotEmpty).toList();
        log('Labels loaded (${_labels?.length}) from assets');
      } catch (_) {}
    }
  }

  List<String>? get labels => _labels;

  void close() {
    try {
      _interpreter?.close();
      _interpreter = null;
    } catch (_) {}
  }

  /// Run inference given a Float32List tensor shaped [1,320,320,3] flattened in row-major order
  List<List<Object>> runInferenceFromTensor(Float32List tensor) {
    if (_interpreter == null) throw Exception('Interpreter not initialized');

    // The interpreter accepts a typed buffer for the input tensor. We must provide an object
    // matching the input shape. A Float32List is acceptable. Provide as single-element list for runForMultipleInputs.
    // If runForMultipleInputs doesn't accept Float32List for your tflite_flutter version, you can wrap as [tensor.buffer.asFloat32List()] or build nested structure.
    final input = tensor; // flattened [320*320*3], interpreter should interpret by shape

    final output = {
      0: [List<double>.filled(10, 0.0)], // scores
      1: [List<List<double>>.filled(10, List<double>.filled(4, 0.0))], // boxes
      2: [0.0], // num detections
      3: [List<double>.filled(10, 0.0)], // classes
    };

    try {
      // runForMultipleInputs accepts a List where each input is a typed buffer matching the input tensor
      List<List<List<List<double>>>> input = List.generate(
        1,
            (_) => List.generate(
          320,
              (y) => List.generate(
            320,
                (x) {
              final base = (y * 320 + x) * 3;
              return [
                tensor[base + 0],
                tensor[base + 1],
                tensor[base + 2],
              ];
            },
          ),
        ),
      );
      _interpreter!.runForMultipleInputs([input], output);
    } catch (e) {
      // Fallback: some tflite_flutter versions expect nested Lists; if error occurs, rethrow after debug
      log('Interpreter run error: $e');
      rethrow;
    }

    return output.values.toList();
  }
}

/// Top-level function for compute() that constructs Float32List directly from CameraImage planes.
///
/// params Map keys:
///  - width (int) : source image width
///  - height (int): source image height
///  - p0 (Uint8List) : Y plane bytes
///  - p1 (Uint8List) : U plane bytes
///  - p2 (Uint8List) : V plane bytes
///  - row0 (int) : bytesPerRow for Y
///  - row1 (int) : bytesPerRow for U
///  - pixelStride1 (int) : bytesPerPixel for U plane (often 2 for NV21)
///
/// Strategy: nearest-neighbor downsample from source -> 320x320 while converting YUV->RGB on the fly.
/// This avoids allocating an intermediate full-size RGB image and dramatically reduces work/copies.
Map<String, dynamic> runConvertCameraImageToMatrix(Map<String, dynamic> params) {
  final int srcW = params['width'] as int;
  final int srcH = params['height'] as int;

  final Uint8List p0 = params['p0'] as Uint8List; // Y
  final Uint8List p1 = params['p1'] as Uint8List; // U
  final Uint8List p2 = params['p2'] as Uint8List; // V

  final int row0 = params['row0'] as int;
  final int row1 = params['row1'] as int;
  final int pixelStride1 = params['pixelStride1'] as int;

  const int dstSize = 320;
  final Float32List tensor = Float32List(dstSize * dstSize * 3);
  int ti = 0;

  // Precompute ratios to map dst -> src (nearest neighbor)
  final double xRatio = srcW / dstSize;
  final double yRatio = srcH / dstSize;

  for (int y = 0; y < dstSize; y++) {
    final int srcY = (y * yRatio).toInt().clamp(0, srcH - 1);
    final int yRow = srcY * row0;
    final int uvRow = (srcY >> 1) * row1;
    for (int x = 0; x < dstSize; x++) {
      final int srcX = (x * xRatio).toInt().clamp(0, srcW - 1);

      // compute indices for Y and UV
      final int yIndex = yRow + srcX;
      final int uvIndex = uvRow + (srcX >> 1) * pixelStride1;

      final int yp = p0[yIndex];
      final int up = p1[uvIndex];
      final int vp = p2[uvIndex];

      // YUV to RGB
      int r = (yp + (1.370705 * (vp - 128))).round();
      int g = (yp - (0.337633 * (up - 128)) - (0.698001 * (vp - 128))).round();
      int b = (yp + (1.732446 * (up - 128))).round();

      if (r < 0) r = 0;
      else if (r > 255) r = 255;
      if (g < 0) g = 0;
      else if (g > 255) g = 255;
      if (b < 0) b = 0;
      else if (b > 255) b = 255;

      // normalize to [-1,1] (model expects that)
      tensor[ti++] = (r / 127.5) - 1.0;
      tensor[ti++] = (g / 127.5) - 1.0;
      tensor[ti++] = (b / 127.5) - 1.0;
    }
  }

  return {'tensor': tensor};
}