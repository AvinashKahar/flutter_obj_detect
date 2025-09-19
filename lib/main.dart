import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:image_picker/image_picker.dart';
import 'video_detection_page.dart';
import 'object_detection.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange),
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

  @override
  void initState() {
    super.initState();
    SchedulerBinding.instance.addPostFrameCallback((_) async {
      objectDetection = ObjectDetection();
      await objectDetection!
          .loadModelFromFile("/storage/emulated/0/Download/ssd_weapon_model_v2.tflite");
      await objectDetection!
          .loadLabelsFromFile("/storage/emulated/0/Download/labels.txt");
    });
  }

  @override
  Widget build(BuildContext context) {
    return const VideoDetectionPage();
  }
}