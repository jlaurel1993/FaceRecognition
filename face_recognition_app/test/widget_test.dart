import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:face_recognition_app/main.dart'; // Ensure this matches your package name

void main() {
  testWidgets('Face Recognition App UI Test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const FaceRecognitionApp());

    // Verify the initial UI elements.
    expect(find.text("Face Recognition App"), findsOneWidget);
    expect(find.text("No image selected"), findsOneWidget);
    expect(find.text("Detected: No face detected"), findsOneWidget);
    expect(find.byType(ElevatedButton), findsOneWidget);

    // Simulate tapping the "Capture Image" button.
    await tester.tap(find.byType(ElevatedButton));
    await tester.pump();

    // Since we can't actually test the camera, just check if the button works.
    expect(find.byType(ElevatedButton), findsOneWidget);
  });
}
