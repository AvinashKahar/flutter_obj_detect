// lib/detection_painter.dart
import 'package:flutter/material.dart';
import 'object_detection.dart';

class DetectionPainter extends CustomPainter {
  final List<DetectionResult> detections;
  DetectionPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()..color = Colors.green..style = PaintingStyle.stroke..strokeWidth = 3;
    final textStyle = const TextStyle(color: Colors.green, fontSize: 14, backgroundColor: Colors.white);

    for (var d in detections) {
      final rect = Rect.fromLTRB(
        d.rect.left * size.width,
        d.rect.top * size.height,
        d.rect.right * size.width,
        d.rect.bottom * size.height,
      );
      canvas.drawRect(rect, paint);
      final tp = TextPainter(
        text: TextSpan(text: '${d.label} ${(d.confidence * 100).toStringAsFixed(1)}%', style: textStyle),
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      tp.paint(canvas, rect.topLeft);
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) => oldDelegate.detections != detections;
}