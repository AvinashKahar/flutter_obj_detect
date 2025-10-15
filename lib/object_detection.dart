// lib/object_detection.dart
import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_obj_detect/file_logger.dart';


/// DetectionResult for UI painting
class DetectionResult {
  final Rect rect; // normalized [0..1]
  final String label;
  final double confidence;
  DetectionResult(this.rect, this.label, this.confidence);
}

class ObjectDetection {
  Interpreter? _interpreter;
  List<String>? _labels;
  String _delegateUsed = 'none';

  ObjectDetection();

  String get delegateUsed => _delegateUsed;

  /// Try to load interpreter from external file and select best delegate available.
  Future<void> loadModelFromFile(String path) async {
    log('loadModelFromFile path $path');

    final f = File(path);
    if (!f.existsSync()) {
      FileLogger.log("Model file not found at $path");
      throw Exception('Model file not found at $path');
    }

    // We will attempt to use NNAPI first, then GPU, then XNNPACK.
    // Note: using multiple delegates at once may conflict; we attempt in order.
    // We create options for each attempt separately.
    Exception? lastError;

    // 1) Try NNAPI (Android only)
    if (Platform.isAndroid) {
      try {
        final opts = InterpreterOptions();
        opts.useNnApiForAndroid = true;
        _interpreter = await Interpreter.fromFile(f, options: opts);
        _delegateUsed = 'NNAPI';
        FileLogger.log("TFModel/Interpreter loaded with NNAPI delegate. from $path");
        log('Interpreter loaded with NNAPI delegate.');
        return;
      } catch (e) {
        lastError = e as Exception;
        FileLogger.log("NNAPI load failed: $e");
        log('NNAPI load failed: $e');
      }
    }

    // 2) Try GPU delegate
    // try {
    //   final opts = InterpreterOptions();
    //   try {
    //     // GpuDelegate is available in tflite_flutter package
    //     opts.addDelegate(GpuDelegate());
    //   } catch (e) {
    //     FileLogger.log("GpuDelegate not available on this platform/version: $e");
    //     log('GpuDelegate not available on this platform/version: $e');
    //   }
    //   _interpreter = await Interpreter.fromFile(f, options: opts);
    //   _delegateUsed = 'GPU';
    //   FileLogger.log("Interpreter loaded with GPU delegate.");
    //   log('Interpreter loaded with GPU delegate.');
    //   return;
    // } catch (e) {
    //   lastError = e as Exception?;
    //   FileLogger.log("GPU load failed: $e");
    //   log('GPU load failed: $e');
    // }
    //
    // // 3) Fallback to XNNPack (CPU optimized)
    // try {
    //   final opts = InterpreterOptions();
    //   opts.addDelegate(XNNPackDelegate());
    //   _interpreter = await Interpreter.fromFile(f, options: opts);
    //   _delegateUsed = 'XNNPACK';
    //   FileLogger.log("Interpreter loaded with XNNPACK delegate.");
    //   log('Interpreter loaded with XNNPACK delegate.');
    //   return;
    // } catch (e) {
    //   lastError = e as Exception?;
    //   FileLogger.log("XNNPACK load failed: $e");
    //   log('XNNPACK load failed: $e');
    // }

    // 4) Try asset fallback (no delegate)
    // try {
    //   final opts = InterpreterOptions();
    //   _interpreter = await Interpreter.fromAsset('assets/ssd_weapon_model_v2.tflite', options: opts);
    //   _delegateUsed = 'AssetFallback';
    //   FileLogger.log("Interpreter loaded from asset fallback (no delegate).");
    //   log('Interpreter loaded from asset fallback (no delegate).');
    //   return;
    // } catch (e) {
    //   lastError = e as Exception?;
    //   FileLogger.log("Asset fallback failed: $e");
    //   log('Asset fallback failed: $e');
    // }

    FileLogger.log("Failed to load interpreter. Last error: $lastError");
    throw Exception('Failed to load interpreter. Last error: $lastError');
  }

  Future<void> loadLabelsFromFile(String path) async {
    log('loadLabelsFromFile path $path');

    final f = File(path);
    if (!f.existsSync()) {
      FileLogger.log("Labels file not found at $path (you can keep labels in assets).");
      log('Labels file not found at $path (you can keep labels in assets).');
      return;
    }
    final raw = await f.readAsString();
    _labels = raw.split('\n').where((s) => s.trim().isNotEmpty).toList();
    FileLogger.log("Labels loaded (${_labels?.length}) from $path");
    log('Labels loaded (${_labels?.length}) from $path');
  }

  /// If interpreter not loaded from file, fallback to asset
  // Future<void> ensureInterpreterInitialized() async {
  //   if (_interpreter == null) {
  //     final opts = InterpreterOptions();
  //     try {
  //       opts.addDelegate(XNNPackDelegate());
  //     } catch (_) {}
  //     _interpreter = await Interpreter.fromAsset('assets/ssd_weapon_model_v2.tflite', options: opts);
  //     _delegateUsed = 'AssetFallback';
  //     FileLogger.log("Interpreter loaded from asset fallback.");
  //     log('Interpreter loaded from asset fallback.');
  //   }
  //   if (_labels == null) {
  //     try {
  //       final raw = await rootBundle.loadString('assets/labels.txt');
  //       _labels = raw.split('\n').where((s) => s.trim().isNotEmpty).toList();
  //       FileLogger.log("Labels loaded (${_labels?.length}) from assets.");
  //       log('Labels loaded (${_labels?.length}) from assets.');
  //     } catch (_) {}
  //   }
  // }

  List<String>? get labels => _labels;

  void close() {
    try {
      _interpreter?.close();
      _interpreter = null;
    } catch (_) {}
  }

  /// Accept a flattened Float32List [320*320*3] and perform inference.
  /// Internally we convert to nested shape [1][320][320][3] as tflite_flutter requires.
  List<List<Object>> runInferenceFromTensor(Float32List tensor) {
    if (_interpreter == null) {
      FileLogger.log("Interpreter not initialized");
      throw Exception('Interpreter not initialized');
    }

    // Convert flattened Float32List into nested List structure [1][H][W][C] (double)
    // Note: conversion cost exists but is small relative to interpreter runtime on your device.
    const int H = 320;
    const int W = 320;
    const int C = 3;
    int idx = 0;

    final List<List<List<List<double>>>> input = List.generate(1, (_) {
      return List.generate(H, (y) {
        return List.generate(W, (x) {
          final r = tensor[idx++].toDouble();
          final g = tensor[idx++].toDouble();
          final b = tensor[idx++].toDouble();
          return <double>[r, g, b];
        });
      });
    });

    final output = {
      0: [List<double>.filled(10, 0.0)],
      1: [List<List<double>>.filled(10, List<double>.filled(4, 0.0))],
      2: [0.0],
      3: [List<double>.filled(10, 0.0)],
    };

    _interpreter!.runForMultipleInputs([input], output);
    return output.values.toList();
  }
}



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