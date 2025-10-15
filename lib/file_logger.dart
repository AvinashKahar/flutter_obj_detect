import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

class FileLogger {
  static final String _fileName = "object_detected_log.txt";
  static File? _logFile;

  /// Initializes the log file
  static Future<void> init() async {
    // Request permission
    var status = await Permission.manageExternalStorage.request();
    if (!status.isGranted) {
      debugPrint("Storage permission not granted");
      return;
    }

    String filePath = "/storage/emulated/0/Download/$_fileName";
    _logFile = File(filePath);

    // Create the file if it doesn't exist
    if (!await _logFile!.exists()) {
      await _logFile!.create(recursive: true);
    }

    debugPrint("Log file initialized at: $filePath");
  }

  /// Appends a log entry to the file
  static Future<void> log(String message) async {
    if (_logFile == null) {
      await init(); // Ensure initialization
    }

    final timestamp = DateTime.now().toIso8601String();
    final logEntry = "[$timestamp] $message\n";

    await _logFile!.writeAsString(logEntry, mode: FileMode.append);
  }
}
