import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart'; // compute()
import 'package:camera/camera.dart';
import 'detection_painter.dart';
import 'object_detection.dart';

class VideoDetectionPage extends StatefulWidget {
  const VideoDetectionPage({super.key});

  @override
  State<VideoDetectionPage> createState() => _VideoDetectionPageState();
}

class _VideoDetectionPageState extends State<VideoDetectionPage> {
  CameraController? _cameraController;
  final ObjectDetection _objectDetection = ObjectDetection();
  List<DetectionResult> _detections = [];
  bool _isDetecting = false;
  DateTime _lastDetectionTime = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    await _objectDetection.ensureInterpreterInitialized();

    final cameras = await availableCameras();
    final camera = cameras.first;
    _cameraController =
        CameraController(camera, ResolutionPreset.medium, enableAudio: false);

    await _cameraController!.initialize();
    _cameraController!.startImageStream((CameraImage image) async {
      final now = DateTime.now();
      if (_isDetecting ||
          now.difference(_lastDetectionTime).inMilliseconds < 5000) return;

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

        final startTime = DateTime.now().millisecondsSinceEpoch;
        final result =
        await compute(runConvertCameraImageToMatrix, params); // isolate
        final Float32List tensor = result['tensor'];
        final output = _objectDetection.runInferenceFromTensor(tensor);
        final endTime = DateTime.now().millisecondsSinceEpoch;
        debugPrint("Inference time: ${endTime - startTime} ms");

        final scores = output[0].first as List<double>;
        final boxes = output[1].first as List<List<double>>;
        final numDetections = (output[2].first as num).toInt();
        final classes = output[3].first as List<double>;

        final newDetections = <DetectionResult>[];
        for (int i = 0; i < numDetections; i++) {
          if (scores[i] > 0.4) {
            final cls = classes[i].toInt();
            final label =
            (_objectDetection.labels != null && cls < _objectDetection.labels!.length)
                ? _objectDetection.labels![cls]
                : "N/A";

            final box = boxes[i];
            final rect = Rect.fromLTRB(
              box[1], // xmin
              box[0], // ymin
              box[3], // xmax
              box[2], // ymax
            );
            newDetections.add(DetectionResult(rect, label, scores[i]));
          }
        }

        if (mounted) {
          setState(() {
            _detections = newDetections;
          });
        }
        debugPrint("Detections: ${newDetections.length} objects");
      } catch (e) {
        debugPrint("Error in detection: $e");
      } finally {
        _isDetecting = false;
      }
    });

    setState(() {});
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
          CustomPaint(painter: DetectionPainter(_detections)),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _objectDetection.close();
    super.dispose();
  }
}