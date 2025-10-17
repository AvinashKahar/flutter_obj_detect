// lib/video_detection_page.dart
// --- small updates to use Uint8List tensor (uint8) ---
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart'; // compute
import 'package:camera/camera.dart';
import 'object_detection.dart';
import 'detection_painter.dart';
import 'package:flutter_obj_detect/file_logger.dart';

class VideoDetectionPage extends StatefulWidget {
  final ObjectDetection objectDetection;
  const VideoDetectionPage({super.key, required this.objectDetection});

  @override
  State<VideoDetectionPage> createState() => _VideoDetectionPageState();
}

class _VideoDetectionPageState extends State<VideoDetectionPage> {
  String TAG = "#flutter_obj_detect/video_detection_page";
  CameraController? _cameraController;
  bool _isDetecting = false;
  List<DetectionResult> _detections = [];

  // 🔹 Added: timestamp for frame throttle
  DateTime? _lastInferenceTime;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final camera = cameras.first;

    _cameraController = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
    await _cameraController!.initialize();

    // set stream
    await _cameraController!.startImageStream((CameraImage img) async {
      if (_isDetecting) return;
      _isDetecting = true;


      // 🔹 Added: frame rate throttle (~1 inference every 400ms)
      final now = DateTime.now();
      if (_lastInferenceTime != null && now.difference(_lastInferenceTime!) < const Duration(milliseconds: 1000)) {
        _isDetecting = false;
        return; // skip this frame to keep preview smooth
      }
      _lastInferenceTime = now;


      try {
        // Convert camera Image via compute (this map returns Uint8List now)
        final preprocessStart = DateTime.now().millisecondsSinceEpoch;

        final params = <String, dynamic>{
          'width': img.width,
          'height': img.height,
          'p0': img.planes[0].bytes,
          'p1': img.planes[1].bytes,
          'p2': img.planes[2].bytes,
          'row0': img.planes[0].bytesPerRow,
          'row1': img.planes[1].bytesPerRow,
          'pixelStride1': img.planes[1].bytesPerPixel ?? 1
        };

        final preResult = await compute(runConvertCameraImageToMatrix, params);
        final preprocessEnd = DateTime.now().millisecondsSinceEpoch;

        final Uint8List tensor = preResult['tensor'] as Uint8List;

        final inferenceStart = DateTime.now().millisecondsSinceEpoch;
        final output = widget.objectDetection.runInferenceFromTensor(tensor);
        final inferenceEnd = DateTime.now().millisecondsSinceEpoch;

        final preprocessMs = preprocessEnd - preprocessStart;
        final inferenceMs = inferenceEnd - inferenceStart;
        final totalMs = inferenceEnd - preprocessStart;

        debugPrint("$TAG Delegate used: ${widget.objectDetection.delegateUsed}");
        debugPrint("$TAG Preprocess: ${preprocessMs} ms | Inference: ${inferenceMs} ms | Total: ${totalMs} ms}");
        FileLogger.log("Delegate used: ${widget.objectDetection.delegateUsed}");
        FileLogger.log("Preprocess: ${preprocessMs} ms | Inference: ${inferenceMs} ms | Total: ${totalMs} ms}");

        // output is remapped to: [scores, boxes, numDetections, classes]
        final scores = (output[0] as dynamic).isNotEmpty
            ? (output[0] as List).first as List<double>
            : <double>[];
        final boxes = (output[1] as List).first as List<List<double>>;
        final numDetections = ((output[2] as List).first as num).toInt();
        final classes = (output[3] as List).first as List<double>;

        final List<DetectionResult> results = [];
        for (int i = 0; i < numDetections; i++) {
          final score = (i < scores.length) ? scores[i] : 0.0;
          if (score < 0.4) continue; // threshold; tune as needed
          final cls = classes[i].toInt();
          final label = (widget.objectDetection.labels != null && cls < widget.objectDetection.labels!.length)
              ? widget.objectDetection.labels![cls]
              : 'class_$cls';
          final b = boxes[i];
          // b format assumed [ymin, xmin, ymax, xmax]
          final rect = Rect.fromLTRB(b[1], b[0], b[3], b[2]);
          results.add(DetectionResult(rect, label, score));
        }

        for(var data in results){
          debugPrint('$TAG Detection: ${data.label} ${(data.confidence * 100).toStringAsFixed(1)}% ${data.rect}');
          FileLogger.log("$TAG Detection: ${data.label} ${(data.confidence * 100).toStringAsFixed(1)}% ${data.rect}");
        }

        debugPrint('$TAG Detections: ${results.length}');
        FileLogger.log("Detections: ${results.length}");

        if (mounted) setState(() => _detections = results);
      } catch (e, st) {
        debugPrint('$TAG Frame processing error: $e\n$st');
        FileLogger.log("Frame processing error: $e\n$st");
      } finally {
        _isDetecting = false;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    return Scaffold(
      body: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(_cameraController!),
            CustomPaint(painter: DetectionPainter(_detections)
            ),
          ]),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }
}