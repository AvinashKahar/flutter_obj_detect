import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_obj_detect/object_detection.dart';
import 'package:flutter_obj_detect/video_detection_page.dart';
import 'package:flutter_obj_detect/file_logger.dart';

class UserInput extends StatefulWidget {
  UserInput({super.key,});

  @override
  State<UserInput> createState() => _UserInputState();
}

class _UserInputState extends State<UserInput> {

  TextEditingController ctrModelController = TextEditingController();
  TextEditingController ctrLabelController = TextEditingController();
  final ObjectDetection objectDetection = ObjectDetection();
  // bool _ready = false;
  // String _status = 'Initializing...';


  @override
  void initState() {
    super.initState();
    SchedulerBinding.instance.addPostFrameCallback((_) async {
      ctrModelController.text = "ssd_weapon_model_v2";
      ctrLabelController.text = "labels";
    });
  }


  @override
  void dispose() {
    objectDetection.close();
    super.dispose();
  }


  initializeInterpreter(String txtModelController, String txtLabelController) async {
    try {
      final modelPath = '/storage/emulated/0/Download/$txtModelController.tflite';
      final labelsPath = '/storage/emulated/0/Download/$txtLabelController.txt';

      // Load interpreter (tries NNAPI/GPU/XNNPACK inside)
      await objectDetection.loadModelFromFile(modelPath);
      await objectDetection.loadLabelsFromFile(labelsPath);
      //await objectDetection.ensureInterpreterInitialized();

      // setState(() {
      //   _ready = true;
      //   _status = 'Ready';
      // });


      Future.delayed(Duration(seconds: 1), () {
        Navigator.push(context, MaterialPageRoute(builder: (context) {
          return VideoDetectionPage(
            objectDetection: objectDetection,
          );
        },));
      });




    } catch (e, st) {
      FileLogger.log("Startup error: $e\n$st");
      debugPrint('Startup error: $e\n$st');
      // setState(() {
      //   _status = 'Init failed: $e';
      // });
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Object Detection'),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 100, vertical: 100),
        child: Column(
          children: [
            TextField(
              controller: ctrModelController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Enter Model Name (without extension)',
              ),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: ctrLabelController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Enter Label Name (without extension)',
              ),
            ),
            const SizedBox(height: 50),
            ElevatedButton(
              onPressed: () async{
                if(ctrLabelController.text.isNotEmpty && ctrModelController.text.isNotEmpty){

                  await initializeInterpreter(ctrModelController.text, ctrLabelController.text);

                  // if(_ready == true){
                  //   Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) {
                  //     return VideoDetectionPage(
                  //       objectDetection: objectDetection,
                  //     );
                  //   },));
                  // }

                }

              },
              child: const Text('Submit'),
            ),
          ],
        ),
      ),
    );
  }
}
