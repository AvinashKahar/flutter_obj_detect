// lib/video_detection_page.dart
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
  CameraController? _cameraController;
  List<DetectionResult> _detections = [];
  bool _isDetecting = false;
  DateTime _lastDetectionTime = DateTime.fromMillisecondsSinceEpoch(0);

  final int throttleMs = 1000; // 1 frame/sec

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    //await widget.objectDetection.ensureInterpreterInitialized();

    final cameras = await availableCameras();
    final camera = cameras.first;
    _cameraController = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
    await _cameraController!.initialize();

    _cameraController!.startImageStream(_processCameraImage);
    if (mounted) setState(() {});
  }

  Future<void> _processCameraImage(CameraImage image) async {
    final now = DateTime.now();
    if (_isDetecting) return;
    if (now.difference(_lastDetectionTime).inMilliseconds < throttleMs) return;

    _isDetecting = true;
    _lastDetectionTime = now;

    try {
      final params = <String, dynamic>{
        'width': image.width,
        'height': image.height,
        'p0': image.planes[0].bytes,
        'p1': image.planes[1].bytes,
        'p2': image.planes[2].bytes,
        'row0': image.planes[0].bytesPerRow,
        'row1': image.planes[1].bytesPerRow,
        'pixelStride1': image.planes[1].bytesPerPixel ?? 1,
      };

      final preprocessStart = DateTime.now().millisecondsSinceEpoch;
      final Map<String, dynamic> preResult = await compute(runConvertCameraImageToMatrix, params);
      final preprocessEnd = DateTime.now().millisecondsSinceEpoch;

      final Float32List tensor = preResult['tensor'] as Float32List;

      final inferenceStart = DateTime.now().millisecondsSinceEpoch;
      final output = widget.objectDetection.runInferenceFromTensor(tensor);
      final inferenceEnd = DateTime.now().millisecondsSinceEpoch;

      final preprocessMs = preprocessEnd - preprocessStart;
      final inferenceMs = inferenceEnd - inferenceStart;
      final totalMs = inferenceEnd - preprocessStart;

      FileLogger.log("Delegate used: ${widget.objectDetection.delegateUsed}");
      FileLogger.log("Preprocess: ${preprocessMs} ms | Inference: ${inferenceMs} ms | Total: ${totalMs} ms}");

      debugPrint('Delegate used: ${widget.objectDetection.delegateUsed}');
      debugPrint('Preprocess: ${preprocessMs} ms | Inference: ${inferenceMs} ms | Total: ${totalMs} ms');

      final scores = output[0].first as List<double>;
      final boxes = output[1].first as List<List<double>>;
      final numDetections = (output[2].first as num).toInt();
      final classes = output[3].first as List<double>;

      final List<DetectionResult> results = [];
      for (int i = 0; i < numDetections; i++) {
        if (scores[i] > 0.4) {
          final cls = classes[i].toInt();
          final label = (widget.objectDetection.labels != null && cls < widget.objectDetection.labels!.length)
              ? widget.objectDetection.labels![cls]
              : 'class_$cls';
          final b = boxes[i];
          final rect = Rect.fromLTRB(b[1], b[0], b[3], b[2]);
          results.add(DetectionResult(rect, label, scores[i]));
        }
      }

      debugPrint('Detections: ${results.length}');
      FileLogger.log("Detections: ${results.length}");

      if (mounted) setState(() => _detections = results);
    } catch (e, st) {
      FileLogger.log("Frame processing error: $e\n$st");
      debugPrint('Frame processing error: $e\n$st');
    } finally {
      _isDetecting = false;
    }
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