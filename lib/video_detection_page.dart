import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart'; // compute
import 'package:flutter_obj_detect/detection_painter.dart';
import 'package:flutter_obj_detect/object_detection.dart';
import 'package:camera/camera.dart';

class VideoDetectionPage extends StatefulWidget {
  const VideoDetectionPage({Key? key}) : super(key: key);

  @override
  _VideoDetectionPageState createState() => _VideoDetectionPageState();
}

class _VideoDetectionPageState extends State<VideoDetectionPage> {
  CameraController? _cameraController;
  ObjectDetection? _objectDetection;
  List<DetectionResult> _detections = [];
  List<DetectionResult> _totalDetection = [];
  bool _isDetecting = false;
  DateTime _lastDetectionTime = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _totalDetection = [];
    _objectDetection = ObjectDetection();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    // Wait for model & labels to be ready
    await _objectDetection!.ensureLoaded();

    final cameras = await availableCameras();
    final camera = cameras.first;

    _cameraController = CameraController(camera, ResolutionPreset.low, enableAudio: false);
    await _cameraController!.initialize();

    _cameraController!.startImageStream((CameraImage image) async {
      // throttle: only run inference once per second (1000 ms)
      final now = DateTime.now();
      if (_isDetecting || now.difference(_lastDetectionTime).inMilliseconds < 2000) return;
      _isDetecting = true;
      _lastDetectionTime = now;

      // prepare data for compute() — keep it serializable
      final params = <String, dynamic>{
        'width': image.width,
        'height': image.height,
        'planes': image.planes.map((p) => p.bytes).toList(),
        'bytesPerRow': image.planes.map((p) => p.bytesPerRow).toList(),
        'bytesPerPixel': image.planes.map((p) => p.bytesPerPixel ?? 1).toList(),
      };

      try {
        // Offload expensive conversion to background isolate
        final List<List<List<num>>> imageMatrix =
        await compute(runConvertCameraImageToMatrix, params);

        // Run inference on main isolate (fast compared to conversion)
        final output = _objectDetection!.runInferenceFromImageMatrix(imageMatrix);

        final scoresTensor = output[0].first as List<double>;
        final boxesTensor = output[1].first as List<List<double>>;
        final classesTensor = output[3].first as List<double>;
        final numberOfDetections = (output[2].first as num).toInt();

        final results = <DetectionResult>[];
        for (int i = 0; i < numberOfDetections; i++) {
          if (scoresTensor[i] > 0.4) {
            final classIndex = classesTensor[i].toInt();
            final label = _objectDetection!.labels![classIndex];

            // Boxes are [ymin, xmin, ymax, xmax] normalized
            final box = boxesTensor[i];
            final rect = Rect.fromLTRB(
              box[1], // xmin
              box[0], // ymin
              box[3], // xmax
              box[2], // ymax
            );

            results.add(DetectionResult(rect, label, scoresTensor[i]));
          }
        }

        if (mounted) {
          setState(() {
            _detections = results;
            _totalDetection = [..._totalDetection, ...results];
            if (_totalDetection.length > 10) {
              _totalDetection = _totalDetection.sublist(_totalDetection.length - 10);
            }
          });
        }
      } catch (e, st) {
        debugPrint('Inference/conversion error: $e\n$st');
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
          Positioned(
            left: 0,
            bottom: 0,
            right: 0,
            child: SizedBox(
              height: 350,
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                itemCount: _totalDetection.length,
                itemBuilder: (context, index) {
                  final d = _totalDetection[index];
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2.0),
                    child: Text(
                      '${index + 1} ${d.label} (${(d.confidence * 100).toStringAsFixed(1)}%)',
                      style: const TextStyle(color: Colors.white, fontSize: 18),
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}