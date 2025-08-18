import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_obj_detect/object_detection.dart';
import 'dart:io' show Platform;
import 'package:camera/camera.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.orange,
        ),
      ),
      home: const MyHome(),
    );
  }
}

class MyHome extends StatefulWidget {
  const MyHome({super.key});

  @override
  State<MyHome> createState() => _MyHomeState();
}

class _MyHomeState extends State<MyHome> {
  final imagePicker = ImagePicker();

  ObjectDetection? objectDetection;

  Uint8List? image;

  @override
  void initState() {
    super.initState();
    objectDetection = ObjectDetection();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // appBar: AppBar(
      //   title: Image.asset('assets/images/tfl_logo.png'),
      //   backgroundColor: Colors.black.withOpacity(0.5),
      // ),
      body: SafeArea(
        child: Column(
          children: <Widget>[
            Expanded(
              child: (image != null) ?
              Container(
                margin: EdgeInsets.all(30),
                child: Image.memory(
                  image!,
                  fit: BoxFit.contain,
                  filterQuality: FilterQuality.high,
                ),
              ) : Container(),
            ),
            SizedBox(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  if (Platform.isAndroid || Platform.isIOS)
                    IconButton(
                      onPressed: () async {
                        final result = await imagePicker.pickImage(
                          source: ImageSource.camera,
                        );
                        if (result != null) {
                          image = objectDetection!.analyseImage(result.path);
                          setState(() {});
                        }
                      },
                      icon: const Icon(
                        Icons.camera,
                        size: 64,
                      ),
                    ),
                  IconButton(
                    onPressed: () async {
                      final result = await imagePicker.pickImage(
                        source: ImageSource.gallery,
                      );
                      if (result != null) {
                        image = objectDetection!.analyseImage(result.path);
                        setState(() {});
                      }
                    },
                    icon: const Icon(
                      Icons.photo,
                      size: 64,
                    ),
                  ),
                  IconButton(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (context) => const VideoDetectionPage(),
                        ),
                      );
                    },
                    icon: const Icon(
                      Icons.videocam,
                      size: 64,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class VideoDetectionPage extends StatefulWidget {
  const VideoDetectionPage({Key? key}) : super(key: key);

  @override
  _VideoDetectionPageState createState() => _VideoDetectionPageState();
}

class _VideoDetectionPageState extends State<VideoDetectionPage> {
  CameraController? _cameraController;
  ObjectDetection? _objectDetection;
  List<DetectionResult> _detections = [];
  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();
    _objectDetection = ObjectDetection();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    _cameraController = CameraController(cameras.first, ResolutionPreset.medium, enableAudio: false);
    await _cameraController!.initialize();

    _cameraController!.startImageStream((CameraImage image) async {
      if (_isDetecting) return;
      _isDetecting = true;

      final results = await _objectDetection!.analyseCameraImage(image);
      if (mounted) {
        setState(() => _detections = results);
      }

      _isDetecting = false;
    });
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),
          CustomPaint(
            painter: DetectionPainter(_detections),
          ),
        ],
      ),
    );
  }
}


class DetectionPainter extends CustomPainter {
  final List<DetectionResult> detections;
  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final textStyle = const TextStyle(
      color: Colors.green,
      fontSize: 14,
      backgroundColor: Colors.white,
    );

    for (var d in detections) {
      // Scale from normalized (0–1) to widget size
      final rect = Rect.fromLTRB(
        d.rect.left * size.width,
        d.rect.top * size.height,
        d.rect.right * size.width,
        d.rect.bottom * size.height,
      );

      canvas.drawRect(rect, paint);

      final tp = TextPainter(
        text: TextSpan(
          text: '${d.label} ${(d.confidence * 100).toStringAsFixed(1)}%',
          style: textStyle,
        ),
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      tp.paint(canvas, rect.topLeft);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}