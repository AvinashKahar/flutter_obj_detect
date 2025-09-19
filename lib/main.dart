// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'video_detection_page.dart';
import 'object_detection.dart';

void main() => runApp(const MyApp());

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final ObjectDetection objectDetection = ObjectDetection();
  bool _ready = false;
  String _status = 'Initializing...';

  @override
  void initState() {
    super.initState();
    SchedulerBinding.instance.addPostFrameCallback((_) async {
      try {
        // Ensure you have runtime permissions granted before these calls.
        final modelPath = '/storage/emulated/0/Download/ssd_weapon_model_v2.tflite';
        final labelsPath = '/storage/emulated/0/Download/labels.txt';

        await objectDetection.loadModelFromFile(modelPath);
        await objectDetection.loadLabelsFromFile(labelsPath);
        await objectDetection.ensureInterpreterInitialized(); // fallback to asset if needed

        setState(() {
          _ready = true;
          _status = 'Ready';
        });
      } catch (e, st) {
        debugPrint('Model/Label load error: $e\n$st');
        setState(() => _status = 'Model/Label load failed: $e');
      }
    });
  }

  @override
  void dispose() {
    objectDetection.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection',
      theme: ThemeData(useMaterial3: true, colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange)),
      home: _ready ? VideoDetectionPage(objectDetection: objectDetection) : Scaffold(body: Center(child: Text(_status))),
    );
  }
}