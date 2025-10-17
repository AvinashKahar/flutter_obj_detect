// lib/object_detection.dart
// --- UPDATED for uint8 EfficientDet-Lite2 ---
import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_obj_detect/file_logger.dart';

/// DetectionResult for UI painting (unchanged)
class DetectionResult {
  final Rect rect; // normalized [0..1]
  final String label;
  final double confidence;
  DetectionResult(this.rect, this.label, this.confidence);
}

class ObjectDetection {
  String TAG = "#flutter_obj_detect/ObjectDetection";
  Interpreter? _interpreter;
  String? _delegateUsed;
  List<String>? _labels;

  String? get delegateUsed => _delegateUsed;
  List<String>? get labels => _labels;

  Future<void> loadModelFromFile(String path) async {
    log('loadModelFromFile path $path');

    final f = File(path);
    if (!f.existsSync()) {
      debugPrint('$TAG Model file not found at $path');
      FileLogger.log("Model file not found at $path");
      throw Exception('Model file not found at $path');
    }

    Exception? lastError;

    // 1) Try NNAPI (Android only)
    if (Platform.isAndroid) {
      try {
        final opts = InterpreterOptions();
        opts.useNnApiForAndroid = true;
        _interpreter = await Interpreter.fromFile(f, options: opts);
        _delegateUsed = 'NNAPI';
        debugPrint('$TAG TFModel/Interpreter loaded with NNAPI delegate. from $path');
        FileLogger.log("TFModel/Interpreter loaded with NNAPI delegate. from $path");
        //log('Interpreter loaded with NNAPI delegate.');
        return;
      } catch (e) {
        lastError = e as Exception;
        debugPrint('$TAG NNAPI load failed: $e');
        FileLogger.log("NNAPI load failed: $e");
        //log('NNAPI load failed: $e');
      }
    }

    // 2) Try GPU delegate (commented in original; user can enable if desired)
    // 3) Try to load with default options (XNNPACK/CPU)
    try {
      final opts = InterpreterOptions();
      _interpreter = await Interpreter.fromFile(f, options: opts);
      _delegateUsed = 'CPU';
      debugPrint('$TAG TFModel/Interpreter loaded on CPU (fallback). from $path');
      FileLogger.log("TFModel/Interpreter loaded on CPU (fallback). from $path");
      //log('Interpreter loaded on CPU fallback.');
      return;
    } catch (e) {
      lastError = e as Exception;
      debugPrint('$TAG CPU fallback load failed: $e');
      FileLogger.log("CPU fallback load failed: $e");
      //log('CPU fallback load failed: $e');
    }

    debugPrint('$TAG Failed to load interpreter. Last error: $lastError');
    FileLogger.log("Failed to load interpreter. Last error: $lastError");
    throw Exception('Failed to load interpreter. Last error: $lastError');
  }

  Future<void> loadLabelsFromFile(String path) async {
    log('loadLabelsFromFile path $path');

    final f = File(path);
    if (!f.existsSync()) {
      debugPrint('$TAG Labels file not found at $path (you can keep labels in assets).');
      FileLogger.log("Labels file not found at $path (you can keep labels in assets).");
      //log('Labels file not found at $path (you can keep labels in assets).');
      return;
    }
    final raw = await f.readAsString();
    _labels = raw.split('\n').where((s) => s.trim().isNotEmpty).toList();
    debugPrint('$TAG Labels loaded (${_labels?.length}) from $path');
    FileLogger.log("Labels loaded (${_labels?.length}) from $path");
    //log('Labels loaded (${_labels?.length}) from $path');
  }

  /// Run inference given a Uint8List tensor produced by runConvertCameraImageToMatrix.
  /// Returns a List in the same format your VideoDetectionPage expects:
  /// [scores, boxes, numDetections, classes]
  List<dynamic> runInferenceFromTensor(Uint8List tensor) {
    if (_interpreter == null) {
      debugPrint('$TAG Interpreter not initialized');
      FileLogger.log("Interpreter not initialized");
      throw Exception('Interpreter not initialized');
    }

    // Build nested input [1][H][W][C] with int values (0..255)
    const int H = 320;
    const int W = 320;
    const int C = 3;
    int idx = 0;
    final input = List.generate(1, (_) {
      return List.generate(H, (y) {
        return List.generate(W, (x) {
          final r = tensor[idx++];
          final g = tensor[idx++];
          final b = tensor[idx++];
          // Many uint8 tflite models expect RGB in [0..255]
          return <int>[r, g, b];
        });
      });
    });

    // Prepare output containers. EfficientDet-Lite exported TFLite commonly has
    // outputs in the order: 0: boxes, 1: classes, 2: scores, 3: num_detections
    // But your app's UI expects the following order:
    // [scores, boxes, numDetections, classes]
    // So we will ask the interpreter for 4 outputs and then remap them below.

    // Ask the interpreter what shape its output tensors actually have
    final boxesShape   = _interpreter!.getOutputTensor(0).shape; // e.g. [1, 25, 4]
    final classesShape = _interpreter!.getOutputTensor(1).shape; // e.g. [1, 25]
    final scoresShape  = _interpreter!.getOutputTensor(2).shape; // e.g. [1, 25]

    final numDetections = boxesShape[1]; // typically 25 for EfficientDet-Lite2

    final outputBoxes   = List.generate(1, (_) =>
        List.generate(numDetections, (_) => List<double>.filled(4, 0.0)));
    final outputClasses = List.generate(1, (_) =>
    List<double>.filled(numDetections, 0.0));
    final outputScores  = List.generate(1, (_) =>
    List<double>.filled(numDetections, 0.0));
    final outputNum = List<double>.filled(1, 0.0);

    final outputs = <int, Object>{
      0: outputBoxes, // expected from model: boxes
      1: outputClasses, // expected from model: classes
      2: outputScores, // expected from model: scores
      3: outputNum, // expected from model: num_detections
    };

    _interpreter!.runForMultipleInputs([input], outputs);

    // Remap to UI expected order: [scores, boxes, numDetections, classes]
    // final remapped = <dynamic>[];
    // remapped.add(outputs[2] as List<List<List<double>>>); // scores
    // remapped.add(outputs[0] as List<List<List<double>>>); // boxes
    // remapped.add(outputs[3] as List<double>); // num detections
    // remapped.add(outputs[1] as List<List<double>>); // classes

    // ---- Fix for 2D vs 3D model output mismatch ----
    final remapped = <dynamic>[];

// Scores tensor: usually [1, N]  → flatten if needed
    var scoresTensor = outputs[2];
    if (scoresTensor is List<List<double>>) {
      remapped.add(scoresTensor); // [1, N]
    } else if (scoresTensor is List<double>) {
      remapped.add([scoresTensor]); // wrap to [1, N]
    } else {
      remapped.add([<double>[]]);
    }

// Boxes tensor: some models give [N,4], others [1,N,4]
    var boxesTensor = outputs[0];
    if (boxesTensor is List<List<List<double>>>) {
      remapped.add(boxesTensor);
    } else if (boxesTensor is List<List<double>>) {
      remapped.add([boxesTensor]); // wrap 2D → 3D
    } else {
      remapped.add([<List<double>>[]]);
    }

// num_detections is always 1D
    remapped.add(outputs[3] as List<double>);

// Classes tensor: usually [1,N] or [N]
    var classesTensor = outputs[1];
    if (classesTensor is List<List<double>>) {
      remapped.add(classesTensor);
    } else if (classesTensor is List<double>) {
      remapped.add([classesTensor]);
    } else {
      remapped.add([<double>[]]);
    }

    return remapped;
  }
}

/// Convert camera image YUV planes -> Uint8List tensor shaped [320*320*3] (RGB)
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
  final Uint8List tensor = Uint8List(dstSize * dstSize * 3);
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

      final int yIndex = yRow + srcX;
      final int uvIndex = uvRow + (srcX >> 1) * pixelStride1;

      final int yp = p0[yIndex];
      final int up = p1[uvIndex];
      final int vp = p2[uvIndex];

      // YUV to RGB integer conversion
      int r = (yp + (1.370705 * (vp - 128))).round();
      int g = (yp - (0.337633 * (up - 128)) - (0.698001 * (vp - 128))).round();
      int b = (yp + (1.732446 * (up - 128))).round();

      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      // For uint8 model: push raw 0..255 values (RGB)
      tensor[ti++] = r;
      tensor[ti++] = g;
      tensor[ti++] = b;
    }
  }

  return {'tensor': tensor};
}