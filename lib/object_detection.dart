import 'dart:developer';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class DetectionResult {
  final Rect rect; // normalized coords
  final String label;
  final double confidence;

  DetectionResult(this.rect, this.label, this.confidence);
}

class ObjectDetection {
  Interpreter? _interpreter;
  List<String>? _labels;

  Future<void> loadModelFromFile(String path) async {
    final f = File(path);
    if (!f.existsSync()) throw Exception("Model file not found: $path");
    final opts = InterpreterOptions();
    if (Platform.isAndroid) opts.addDelegate(XNNPackDelegate());
    if (Platform.isIOS) opts.addDelegate(GpuDelegate());
    _interpreter = await Interpreter.fromFile(f, options: opts);
    log("Model loaded from $path");
  }

  Future<void> loadLabelsFromFile(String path) async {
    final f = File(path);
    if (!f.existsSync()) return;
    final raw = await f.readAsString();
    _labels = raw.split("\n").where((s) => s.trim().isNotEmpty).toList();
    log("Labels loaded: ${_labels?.length}");
  }

  Future<void> ensureInterpreterInitialized() async {
    if (_interpreter == null) {
      final opts = InterpreterOptions();
      if (Platform.isAndroid) opts.addDelegate(XNNPackDelegate());
      if (Platform.isIOS) opts.addDelegate(GpuDelegate());
      //_interpreter = await Interpreter.fromAsset("assets/ssd_weapon_model_v2.tflite", options: opts);
      _interpreter = await Interpreter.fromFile(File("/storage/emulated/0/Download/ssd_weapon_model_v2.tflite"));
      //_interpreter = await Interpreter.fromFile(File("sdcard/Download/ssd_weapon_model_v2.tflite"));

    }
    if (_labels == null) {
      try {
        //final raw = await rootBundle.loadString("assets/labels.txt");
        final raw = await File("/storage/emulated/0/Download/labels.txt").readAsString();
        _labels = raw.split("\n").where((s) => s.trim().isNotEmpty).toList();
      } catch (_) {}
    }
  }

  List<String>? get labels => _labels;

  List<List<Object>> runInferenceFromTensor(Float32List tensor) {
    if (_interpreter == null) throw Exception("Interpreter not ready");
    final input = tensor.reshape([1, 320, 320, 3]);

    final output = {
      0: [List<double>.filled(10, 0.0)], // scores
      1: [List<List<double>>.filled(10, List<double>.filled(4, 0.0))], // boxes
      2: [0.0], // num detections
      3: [List<double>.filled(10, 0.0)], // classes
    };
    _interpreter!.runForMultipleInputs([input], output);
    return output.values.toList();
  }

  void close() {
    _interpreter?.close();
    _interpreter = null;
  }
}

/// Runs inside isolate
Map<String, dynamic> runConvertCameraImageToMatrix(Map<String, dynamic> params) {
  final int width = params['width'];
  final int height = params['height'];
  final Uint8List p0 = params['p0'];
  final Uint8List p1 = params['p1'];
  final Uint8List p2 = params['p2'];
  final int row0 = params['row0'];
  final int row1 = params['row1'];
  final int pixelStride1 = params['pixelStride1'];

  final img.Image rgb = img.Image(width: width, height: height);
  for (int h = 0; h < height; h++) {
    final int uvRow = row1 * (h >> 1);
    final int yRow = row0 * h;
    for (int w = 0; w < width; w++) {
      final int uvIndex = uvRow + (pixelStride1 * (w >> 1));
      final int yIndex = yRow + w;
      final int yp = p0[yIndex];
      final int up = p1[uvIndex];
      final int vp = p2[uvIndex];
      int r = (yp + 1.370705 * (vp - 128)).round();
      int g = (yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128)).round();
      int b = (yp + 1.732446 * (up - 128)).round();
      rgb.setPixelRgb(w, h, r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255));
    }
  }

  final img.Image resized = img.copyResize(rgb, width: 320, height: 320);

  final Float32List input = Float32List(320 * 320 * 3);
  int i = 0;
  for (int y = 0; y < 320; y++) {
    for (int x = 0; x < 320; x++) {
      final p = resized.getPixel(x, y);
      input[i++] = (p.r / 127.5) - 1.0;
      input[i++] = (p.g / 127.5) - 1.0;
      input[i++] = (p.b / 127.5) - 1.0;
    }
  }

  return {'tensor': input};
}