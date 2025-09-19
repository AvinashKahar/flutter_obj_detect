import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_obj_detect/video_detection_page.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_obj_detect/object_detection.dart';



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
    SchedulerBinding.instance.addPostFrameCallback((timeStamp) async{
      objectDetection = ObjectDetection();
      await objectDetection!.loadModelFromFile("sdcard/Download/ssd_weapon_model_v2.tflite");
    },);

    super.initState();

  }

  @override
  Widget build(BuildContext context) {
    return VideoDetectionPage();
  }
}