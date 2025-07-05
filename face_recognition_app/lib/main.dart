import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:dio/dio.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:battery_plus/battery_plus.dart';

void main() {
  runApp(const FaceRecognitionApp());
}

class FaceRecognitionApp extends StatefulWidget {
  const FaceRecognitionApp({super.key});

  @override
  _FaceRecognitionAppState createState() => _FaceRecognitionAppState();
}

class _FaceRecognitionAppState extends State<FaceRecognitionApp> {
  XFile? _image;
  String _recognizedName = "No face detected";
  final ImagePicker _picker = ImagePicker();
  final FlutterTts _flutterTts = FlutterTts();
  final Battery _battery = Battery();
  int _batteryLevel = 100;

  // ✅ Using your ngrok public URL
  final String baseUrl = "https://d100-129-113-10-110.ngrok-free.app";
  late final String recognizeUrl = "$baseUrl/recognize";
  late final String registerUrl = "$baseUrl/register_face";
  late final String knownFacesUrl = "$baseUrl/known_faces";

  @override
  void initState() {
    super.initState();
    _checkBatteryStatus();
  }

  Future<void> _checkBatteryStatus() async {
    int batteryLevel = await _battery.batteryLevel;
    setState(() {
      _batteryLevel = batteryLevel;
    });

    if (batteryLevel <= 20) {
      _speak("Warning! Battery is low, please charge your device.");
      _showLowBatteryWarning();
    }
  }

  void _showLowBatteryWarning() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Low Battery"),
        content: const Text("Battery is below 20%. Please charge your device."),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("OK")),
        ],
      ),
    );
  }

  Future<void> _pickImage(ImageSource source, {bool isRegister = false}) async {
    XFile? picked = await _picker.pickImage(source: source);
    if (picked != null) {
      setState(() {
        _image = picked;
        _recognizedName = isRegister ? "Ready to register..." : "Processing...";
      });
      if (isRegister) {
        _promptForNameAndRegister(File(picked.path));
      } else {
        _sendImageToRecognize(File(picked.path));
      }
    }
  }

  Future<void> _promptForNameAndRegister(File image) async {
    String name = "";
    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Enter Name"),
        content: TextField(
          onChanged: (value) => name = value,
          decoration: const InputDecoration(labelText: "Name"),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text("Cancel")),
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Register"),
          ),
        ],
      ),
    );

    if (name.trim().isEmpty) {
      _speak("Registration cancelled. No name entered.");
      return;
    }

    try {
      FormData formData = FormData.fromMap({
        "name": name,
        "image": await MultipartFile.fromFile(image.path, filename: "register.jpg"),
      });

      final response = await Dio().post(registerUrl, data: formData);
      if (response.statusCode == 200 && response.data["message"] != null) {
        setState(() {
          _recognizedName = "Registered: $name";
        });
        _speak("Face registered for $name.");
      } else {
        setState(() {
          _recognizedName = "Registration failed";
        });
        _speak("Face could not be registered.");
      }
    } catch (e) {
      setState(() => _recognizedName = "Error sending registration request");
      _speak("Error occurred during registration.");
    }
  }

  Future<void> _sendImageToRecognize(File imageFile) async {
    try {
      FormData formData = FormData.fromMap({
        "file": await MultipartFile.fromFile(imageFile.path, filename: "face.jpg"),
      });

      final response = await Dio().post(recognizeUrl, data: formData);
      if (response.statusCode == 200) {
        setState(() => _recognizedName = response.data["name"]);
        _speak("Detected $_recognizedName");
      } else {
        setState(() => _recognizedName = "Error: ${response.statusCode}");
      }
    } catch (e) {
      setState(() => _recognizedName = "Error sending request");
    }
  }

  Future<void> _speak(String message) async {
    await _flutterTts.speak(message);
  }

  Future<void> _getKnownFaces() async {
    try {
      final response = await Dio().get(knownFacesUrl);
      if (response.statusCode == 200) {
        List<dynamic> faces = response.data["faces"];
        _speak("Known faces: ${faces.join(", ")}");
      }
    } catch (e) {
      _speak("Failed to get known faces.");
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text("Face Recognition App")),
        drawer: _buildNavigationDrawer(),
        body: SingleChildScrollView(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: 20),
                BatteryStatusWidget(batteryPercentage: _batteryLevel),
                const SizedBox(height: 10),
                _image != null
                    ? Image.file(File(_image!.path), height: 200)
                    : const Text("No image selected", style: TextStyle(fontSize: 16)),
                const SizedBox(height: 20),
                Text("Detected: $_recognizedName", style: const TextStyle(fontSize: 20)),
                const SizedBox(height: 20),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    ElevatedButton(
                      onPressed: () => _pickImage(ImageSource.camera),
                      child: const Text("Capture Image"),
                    ),
                    ElevatedButton(
                      onPressed: () => _pickImage(ImageSource.gallery),
                      child: const Text("Upload from Gallery"),
                    ),
                    ElevatedButton(
                      onPressed: _getKnownFaces,
                      child: const Text("Get Known Faces"),
                    ),
                    ElevatedButton(
                      onPressed: () => _pickImage(ImageSource.camera, isRegister: true),
                      child: const Text("Register New Face"),
                    ),
                  ],
                ),
                const SizedBox(height: 30),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavigationDrawer() {
    return Drawer(
      child: ListView(
        children: [
          const DrawerHeader(
            decoration: BoxDecoration(color: Colors.blue),
            child: Text("Menu", style: TextStyle(color: Colors.white, fontSize: 24)),
          ),
          ListTile(
            title: const Text("Check Battery"),
            onTap: () {
              Navigator.pop(context);
              _checkBatteryStatus();
            },
          ),
        ],
      ),
    );
  }
}

class BatteryStatusWidget extends StatelessWidget {
  final int batteryPercentage;

  const BatteryStatusWidget({super.key, required this.batteryPercentage});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Text("Battery Status", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        Text("$batteryPercentage%", style: const TextStyle(fontSize: 16)),
        if (batteryPercentage <= 20)
          const Text(
            "⚠️ Low Battery!",
            style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold),
          ),
      ],
    );
  }
}
