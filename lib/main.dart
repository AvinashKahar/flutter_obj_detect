// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_obj_detect/file_logger.dart';
import 'package:flutter_obj_detect/user_input.dart';
import 'video_detection_page.dart';
import 'object_detection.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FileLogger.init();

  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  // final ObjectDetection objectDetection = ObjectDetection();
  // bool _ready = false;
  // String _status = 'Initializing...';
  //
  // @override
  // void initState() {
  //   super.initState();
  //   SchedulerBinding.instance.addPostFrameCallback((_) async {
  //     try {
  //       final modelPath = '/storage/emulated/0/Download/ssd_weapon_model_v2.tflite';
  //       final labelsPath = '/storage/emulated/0/Download/labels.txt';
  //
  //
  //       // Load interpreter (tries NNAPI/GPU/XNNPACK inside)
  //       await objectDetection.loadModelFromFile(modelPath);
  //       await objectDetection.loadLabelsFromFile(labelsPath);
  //       // await objectDetection.ensureInterpreterInitialized();
  //
  //       setState(() {
  //         _ready = true;
  //         _status = 'Ready';
  //       });
  //     } catch (e, st) {
  //       FileLogger.log("Startup error: $e\n$st");
  //       debugPrint('Startup error: $e\n$st');
  //       setState(() {
  //         _status = 'Init failed: $e';
  //       });
  //     }
  //   });
  // }
  //
  // @override
  // void dispose() {
  //   objectDetection.close();
  //   super.dispose();
  // }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Detection (NNAPI/GPU tries)',
      theme: ThemeData(useMaterial3: true, colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange)),
      // home: _ready ? VideoDetectionPage(objectDetection: objectDetection) : Scaffold(body: Center(child: Text(_status))),
      home: UserInput(),
    );
  }
}