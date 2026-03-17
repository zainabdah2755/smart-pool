// ─────────────────────────────────────────────────────────────
// smart_pool_service.dart  —  Flutter API Service Layer
// ─────────────────────────────────────────────────────────────
// Handles all communication with the Smart Pool AI backend.
// Usage in a widget:
//
//   final service = SmartPoolService();
//   final result = await service.predict(SensorPayload(...));
// ─────────────────────────────────────────────────────────────

import 'dart:convert';
import 'package:http/http.dart' as http;

// ─────────────────────
// CONFIGURATION
// ─────────────────────

class ApiConfig {
  // Change this to your server IP when deploying on a real device
  static const String baseUrl = 'https://smart-pool-production.up.railway.app';
  static const Duration timeout = Duration(seconds: 15);
}

// ─────────────────────
// REQUEST MODEL
// ─────────────────────

class SensorPayload {
  final double tempEau;
  final double tempAir;
  final double humidite;
  final double ph;
  final double luminosite;
  final double ir;
  final double poolVolumeM3;
  final double? latitude;
  final double? longitude;
  final double? currentFreeCl;

  const SensorPayload({
    required this.tempEau,
    required this.tempAir,
    required this.humidite,
    required this.ph,
    required this.luminosite,
    required this.ir,
    this.poolVolumeM3 = 50.0,
    this.latitude,
    this.longitude,
    this.currentFreeCl,
  });

  Map<String, dynamic> toJson() => {
        'temp_eau': tempEau,
        'temp_air': tempAir,
        'humidite': humidite,
        'ph': ph,
        'luminosite': luminosite,
        'ir': ir,
        'pool_volume_m3': poolVolumeM3,
        if (latitude != null) 'latitude': latitude,
        if (longitude != null) 'longitude': longitude,
        if (currentFreeCl != null) 'current_free_cl': currentFreeCl,
      };
}

// ─────────────────────
// RESPONSE MODELS
// ─────────────────────

class MaintenanceAlert {
  final String level;    // Info | Warning | Critical
  final String message;
  final String action;

  const MaintenanceAlert({
    required this.level,
    required this.message,
    required this.action,
  });

  factory MaintenanceAlert.fromJson(Map<String, dynamic> j) =>
      MaintenanceAlert(
        level: j['level'] as String,
        message: j['message'] as String,
        action: j['action'] as String,
      );
}

class ChlorineInfo {
  final int doseGrams;
  final double freeChlorineTarget;
  final bool shockRequired;
  final List<String> rationale;

  const ChlorineInfo({
    required this.doseGrams,
    required this.freeChlorineTarget,
    required this.shockRequired,
    required this.rationale,
  });

  factory ChlorineInfo.fromJson(Map<String, dynamic> j) => ChlorineInfo(
        doseGrams: j['dose_grams'] as int,
        freeChlorineTarget: (j['free_chlorine_target'] as num).toDouble(),
        shockRequired: j['shock_required'] as bool,
        rationale: List<String>.from(j['rationale'] as List),
      );
}

class WeatherInfo {
  final double? temperature;
  final double? humidity;
  final double? uvIndex;
  final String? description;
  final double? chlorineExtraPpm;
  final String? evaporationLevel;
  final String? contamination_risk;
  final bool? coverRecommendation;
  final String? maintenanceAdvisory;
  final List<String> forecastSummary;
  final String? error;

  const WeatherInfo({
    this.temperature,
    this.humidity,
    this.uvIndex,
    this.description,
    this.chlorineExtraPpm,
    this.evaporationLevel,
    this.contamination_risk,
    this.coverRecommendation,
    this.maintenanceAdvisory,
    this.forecastSummary = const [],
    this.error,
  });

  factory WeatherInfo.fromJson(Map<String, dynamic> j) => WeatherInfo(
        temperature: (j['temperature'] as num?)?.toDouble(),
        humidity: (j['humidity'] as num?)?.toDouble(),
        uvIndex: (j['uv_index'] as num?)?.toDouble(),
        description: j['description'] as String?,
        chlorineExtraPpm: (j['chlorine_extra_ppm'] as num?)?.toDouble(),
        evaporationLevel: j['evaporation_level'] as String?,
        contamination_risk: j['contamination_risk'] as String?,
        coverRecommendation: j['cover_recommendation'] as bool?,
        maintenanceAdvisory: j['maintenance_advisory'] as String?,
        forecastSummary: j['forecast_summary'] != null
            ? List<String>.from(j['forecast_summary'] as List)
            : const [],
        error: j['error'] as String?,
      );
}

class PoolPrediction {
  final String status;           // Normal | Warning | Danger
  final double healthScore;      // 0-100
  final double algaeRisk;        // 0-100
  final String algaeLabel;       // Low | Moderate | High | Critical
  final ChlorineInfo chlorine;
  final bool needsMaintenance;
  final List<MaintenanceAlert> maintenanceAlerts;
  final List<String> recommendations;
  final WeatherInfo? weather;
  final String modelVersion;

  const PoolPrediction({
    required this.status,
    required this.healthScore,
    required this.algaeRisk,
    required this.algaeLabel,
    required this.chlorine,
    required this.needsMaintenance,
    required this.maintenanceAlerts,
    required this.recommendations,
    this.weather,
    required this.modelVersion,
  });

  factory PoolPrediction.fromJson(Map<String, dynamic> j) => PoolPrediction(
        status: j['status'] as String,
        healthScore: (j['health_score'] as num).toDouble(),
        algaeRisk: (j['algae_risk'] as num).toDouble(),
        algaeLabel: j['algae_label'] as String,
        chlorine: ChlorineInfo.fromJson(j['chlorine'] as Map<String, dynamic>),
        needsMaintenance: j['needs_maintenance'] as bool,
        maintenanceAlerts: (j['maintenance_alerts'] as List)
            .map((e) => MaintenanceAlert.fromJson(e as Map<String, dynamic>))
            .toList(),
        recommendations: List<String>.from(j['recommendations'] as List),
        weather: j['weather'] != null
            ? WeatherInfo.fromJson(j['weather'] as Map<String, dynamic>)
            : null,
        modelVersion: j['model_version'] as String,
      );

  /// Convenience: status color for UI
  String get statusColor {
    switch (status) {
      case 'Normal':  return '#4CAF50';
      case 'Warning': return '#FF9800';
      case 'Danger':  return '#F44336';
      default:        return '#9E9E9E';
    }
  }
}

// ─────────────────────
// SERVICE CLASS
// ─────────────────────

class SmartPoolService {
  final String _baseUrl;
  final http.Client _client;

  SmartPoolService({
    String? baseUrl,
    http.Client? client,
  })  : _baseUrl = baseUrl ?? ApiConfig.baseUrl,
        _client = client ?? http.Client();

  /// Full prediction including optional weather
  Future<PoolPrediction> predict(SensorPayload payload) async {
    final uri = Uri.parse('$_baseUrl/predict');
    final resp = await _client
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload.toJson()),
        )
        .timeout(ApiConfig.timeout);

    if (resp.statusCode == 200) {
      return PoolPrediction.fromJson(
          jsonDecode(resp.body) as Map<String, dynamic>);
    }
    throw ApiException(resp.statusCode, resp.body);
  }

  /// Quick prediction without weather (lower latency)
  Future<PoolPrediction> predictQuick(SensorPayload payload) async {
    final uri = Uri.parse('$_baseUrl/predict/quick');
    final resp = await _client
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload.toJson()),
        )
        .timeout(ApiConfig.timeout);

    if (resp.statusCode == 200) {
      return PoolPrediction.fromJson(
          jsonDecode(resp.body) as Map<String, dynamic>);
    }
    throw ApiException(resp.statusCode, resp.body);
  }

  /// Fetch a simulated sensor payload (for demo/test mode)
  Future<Map<String, dynamic>> simulate({String scenario = 'normal'}) async {
    final uri = Uri.parse('$_baseUrl/simulate?scenario=$scenario');
    final resp = await _client.get(uri).timeout(ApiConfig.timeout);
    if (resp.statusCode == 200) {
      return jsonDecode(resp.body) as Map<String, dynamic>;
    }
    throw ApiException(resp.statusCode, resp.body);
  }

  /// Check API health
  Future<bool> isHealthy() async {
    try {
      final resp = await _client
          .get(Uri.parse('$_baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  void dispose() => _client.close();
}

// ─────────────────────
// EXCEPTION
// ─────────────────────

class ApiException implements Exception {
  final int statusCode;
  final String body;

  const ApiException(this.statusCode, this.body);

  @override
  String toString() => 'ApiException($statusCode): $body';
}

// ─────────────────────────────────────────────────────────────
// EXAMPLE WIDGET USAGE (copy into your Flutter screen)
// ─────────────────────────────────────────────────────────────
//
// class PoolDashboard extends StatefulWidget { ... }
//
// class _PoolDashboardState extends State<PoolDashboard> {
//   final _service = SmartPoolService();
//   PoolPrediction? _result;
//   bool _loading = false;
//
//   Future<void> _refresh() async {
//     setState(() => _loading = true);
//     try {
//       // In production: read from ESP32 via Firebase or BLE
//       // In demo mode: use simulated data
//       final sim = await _service.simulate(scenario: 'warning');
//       final payload = SensorPayload(
//         tempEau:   sim['temp_eau'],
//         tempAir:   sim['temp_air'],
//         humidite:  sim['humidite'],
//         ph:        sim['ph'],
//         luminosite:sim['luminosite'],
//         ir:        sim['ir'],
//         latitude:  31.6295,   // Marrakesh example
//         longitude: -7.9811,
//       );
//       final result = await _service.predict(payload);
//       setState(() { _result = result; _loading = false; });
//     } catch (e) {
//       setState(() => _loading = false);
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text('Error: $e')),
//       );
//     }
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     if (_loading) return const CircularProgressIndicator();
//     if (_result == null) return ElevatedButton(onPressed: _refresh, child: const Text('Load'));
//     return Column(children: [
//       Text('Status: ${_result!.status}'),
//       Text('Health: ${_result!.healthScore}/100'),
//       Text('Algae Risk: ${_result!.algaeRisk} [${_result!.algaeLabel}]'),
//       Text('Chlorine: ${_result!.chlorine.doseGrams}g'),
//       ...(_result!.maintenanceAlerts.map((a) => Text('[${a.level}] ${a.message}'))),
//     ]);
//   }
// }
