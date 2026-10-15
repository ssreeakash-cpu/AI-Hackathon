import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default Leaflet icon assets
import L from 'leaflet';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

export default function App() {
  const [activeTab, setActiveTab] = useState('satellite'); // 'satellite', 'camera', 'temporal'
  const [region, setRegion] = useState('india'); // 'india' or 'global'
  const [nasaData, setNasaData] = useState({ hotspots: [] });
  const [cameraSamples, setCameraSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState('');
  const [cameraInference, setCameraInference] = useState(null);
  const [temporalData, setTemporalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mapCenter, setMapCenter] = useState([20.5937, 78.9629]); // India center default
  const [mapZoom, setMapZoom] = useState(5);

  // Fetch camera samples on mount
  useEffect(() => {
    fetch('http://localhost:8001/camera_samples')
      .then(res => res.json())
      .then(samples => {
        setCameraSamples(samples);
        if (samples.length > 0) setSelectedSample(samples[0].filename);
      })
      .catch(err => console.log("Failed to load camera samples:", err));

    // Load India satellite data by default
    loadNasaSatelliteData('india');
  }, []);

  const loadNasaSatelliteData = async (targetRegion) => {
    setLoading(true);
    setError(null);
    setRegion(targetRegion);
    try {
      const res = await fetch(`http://localhost:8001/nasa_hotspots?region=${targetRegion}&limit=80`);
      if (!res.ok) throw new Error("Could not reach backend on Port 8001.");
      const data = await res.json();
      setNasaData(data);
      if (data.hotspots && data.hotspots.length > 0) {
        setMapCenter([data.hotspots[0].latitude, data.hotspots[0].longitude]);
        setMapZoom(targetRegion === 'india' ? 5 : 3);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runCameraInference = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = selectedSample 
        ? `http://localhost:8001/predict/image?sample_name=${selectedSample}`
        : 'http://localhost:8001/predict/image';
      const res = await fetch(url, { method: 'POST' });
      if (!res.ok) throw new Error("Backend server error during inference.");
      const result = await res.json();
      setCameraInference(result);
      if (result.latitude && result.longitude) {
        setMapCenter([result.latitude, result.longitude]);
        setMapZoom(9);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadTemporalAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8001/satellite/temporal_analysis');
      if (!res.ok) throw new Error("Could not load ConvLSTM temporal analysis.");
      const result = await res.json();
      setTemporalData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ backgroundColor: '#121212', color: '#f5f6fa', minHeight: '100vh', padding: '24px', fontFamily: 'Segoe UI, Roboto, system-ui, sans-serif' }}>
      
      {/* Header */}
      <header style={{ borderBottom: '1px solid #2f3640', paddingBottom: '16px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: 0, color: '#e84118', fontSize: '26px' }}>🔥 AI Wildfire Early-Warning Command Center</h1>
            <p style={{ margin: '6px 0 0 0', color: '#718093', fontSize: '14px' }}>
              Problem Statement 25 • VIT Tech-a-thon • Real NASA Earthdata (MODIS & VIIRS) + Spatial-Temporal ConvLSTM
            </p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <span style={{ background: '#273c75', padding: '6px 12px', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold' }}>
              REAL NASA DATASET: ACTIVE
            </span>
            <span style={{ background: '#44bd32', padding: '6px 12px', borderRadius: '4px', fontSize: '12px', fontWeight: 'bold' }}>
              PYTORCH ENGINE: READY
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '12px', marginTop: '18px' }}>
          <button 
            onClick={() => { setActiveTab('satellite'); loadNasaSatelliteData(region); }}
            style={{ 
              padding: '10px 20px', 
              background: activeTab === 'satellite' ? '#e84118' : '#2f3640', 
              color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' 
            }}>
            🛰️ NASA Satellite Hotspots (MODIS / VIIRS)
          </button>
          
          <button 
            onClick={() => { setActiveTab('camera'); runCameraInference(); }}
            style={{ 
              padding: '10px 20px', 
              background: activeTab === 'camera' ? '#e84118' : '#2f3640', 
              color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' 
            }}>
            📹 Camera Surveillance (Smoke vs Fog/Clouds)
          </button>

          <button 
            onClick={() => { setActiveTab('temporal'); loadTemporalAnalysis(); }}
            style={{ 
              padding: '10px 20px', 
              background: activeTab === 'temporal' ? '#e84118' : '#2f3640', 
              color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' 
            }}>
            ⏳ Spatial-Temporal ConvLSTM (NetCDF Cubes)
          </button>
        </div>
      </header>

      {error && (
        <div style={{ padding: '12px', background: '#c23616', borderRadius: '6px', marginBottom: '16px' }}>
          ⚠️ {error}
        </div>
      )}

      {/* Main Content Area */}
      <div style={{ display: 'flex', gap: '20px' }}>
        
        {/* Left Side Controls & Telemetry */}
        <div style={{ flex: 1, background: '#1e272e', padding: '20px', borderRadius: '8px', maxHeight: '680px', overflowY: 'auto' }}>
          
          {/* TAB 1: SATELLITE CONTROLS */}
          {activeTab === 'satellite' && (
            <div>
              <h3 style={{ margin: '0 0 12px 0', color: '#f5f6fa', borderBottom: '1px solid #485460', paddingBottom: '8px' }}>
                NASA Satellite Observations
              </h3>
              
              <div style={{ marginBottom: '16px' }}>
                <label style={{ fontSize: '13px', color: '#d2dae2', display: 'block', marginBottom: '6px' }}>Select Geographical Coverage:</label>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button 
                    onClick={() => loadNasaSatelliteData('india')}
                    disabled={loading}
                    style={{ 
                      flex: 1, padding: '8px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontWeight: 'bold',
                      background: region === 'india' ? '#00a8ff' : '#485460', color: 'white' 
                    }}>
                    🇮🇳 India & South Asia
                  </button>
                  <button 
                    onClick={() => loadNasaSatelliteData('global')}
                    disabled={loading}
                    style={{ 
                      flex: 1, padding: '8px', borderRadius: '4px', border: 'none', cursor: 'pointer', fontWeight: 'bold',
                      background: region === 'global' ? '#00a8ff' : '#485460', color: 'white' 
                    }}>
                    🌍 Global Wildfires
                  </button>
                </div>
              </div>

              <div style={{ fontSize: '13px', lineHeight: '1.6', background: '#2d3436', padding: '12px', borderRadius: '6px', marginBottom: '14px' }}>
                <p style={{ margin: '0' }}><strong>Sensor 1:</strong> NASA VIIRS 375m (Suomi-NPP / NOAA-20)</p>
                <p style={{ margin: '0' }}><strong>Sensor 2:</strong> NASA MODIS 1km (Terra & Aqua)</p>
                <p style={{ margin: '0' }}><strong>Active Observations Loaded:</strong> {nasaData.hotspots ? nasaData.hotspots.length : 0}</p>
                <p style={{ margin: '0' }}><strong>Data Source:</strong> Real NASA Earthdata FIRMS CSVs</p>
              </div>

              <h4 style={{ margin: '14px 0 8px 0', color: '#ff7675' }}>Active Telemetry Table</h4>
              <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                {nasaData.hotspots && nasaData.hotspots.slice(0, 8).map((spot, idx) => (
                  <div key={idx} style={{ background: '#2f3542', padding: '10px', borderRadius: '6px', marginBottom: '8px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <strong style={{ color: '#ffa502' }}>{spot.source}</strong>
                      <span style={{ color: '#70a1ff' }}>FRP: {spot.frp_mw.toFixed(1)} MW</span>
                    </div>
                    <div>Lat: {spot.latitude.toFixed(4)}, Lon: {spot.longitude.toFixed(4)}</div>
                    <div>Brightness: {spot.brightness_kelvin.toFixed(1)} K | Conf: {spot.confidence}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* TAB 2: CAMERA SURVEILLANCE */}
          {activeTab === 'camera' && (
            <div>
              <h3 style={{ margin: '0 0 12px 0', color: '#f5f6fa', borderBottom: '1px solid #485460', paddingBottom: '8px' }}>
                Camera Surveillance & Distractor AI
              </h3>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ fontSize: '13px', color: '#d2dae2', display: 'block', marginBottom: '6px' }}>Select Camera Benchmark Sample:</label>
                <select 
                  value={selectedSample} 
                  onChange={(e) => setSelectedSample(e.target.value)}
                  style={{ width: '100%', padding: '8px', background: '#2f3542', color: 'white', border: '1px solid #57606f', borderRadius: '4px' }}>
                  {cameraSamples.map((s, idx) => (
                    <option key={idx} value={s.filename}>{s.filename}</option>
                  ))}
                </select>
                <button 
                  onClick={runCameraInference}
                  disabled={loading}
                  style={{ marginTop: '10px', width: '100%', padding: '10px', background: '#0984e3', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
                  {loading ? 'Evaluating Model...' : 'Run PyTorch Model Inference'}
                </button>
              </div>

              {cameraInference && (
                <div style={{ fontSize: '13px', lineHeight: '1.6', background: '#2d3436', padding: '14px', borderRadius: '6px' }}>
                  <h4 style={{ margin: '0 0 10px 0', color: '#7bed9f' }}>Model Output Telemetry</h4>
                  <p style={{ margin: '0' }}><strong>Classification:</strong> <span style={{ color: '#ff4757', fontWeight: 'bold' }}>{cameraInference.predicted_class}</span></p>
                  <p style={{ margin: '0' }}><strong>Smoke Confidence:</strong> {(cameraInference.smoke_confidence * 100).toFixed(1)}%</p>
                  <p style={{ margin: '0' }}><strong>Fire Confidence:</strong> {(cameraInference.fire_confidence * 100).toFixed(1)}%</p>
                  <p style={{ margin: '0' }}><strong>Distractor Check (Clouds/Fog/Dust):</strong> {cameraInference.distractor_detected ? '⚠️ Visually similar clouds/fog present' : '✅ Clear of ambient clouds'}</p>
                  <p style={{ margin: '0' }}><strong>Severity Calculation:</strong> <span style={{ color: '#e67e22', fontWeight: 'bold' }}>{cameraInference.severity}</span></p>
                  <p style={{ margin: '0' }}><strong>Multi-Model Risk Score:</strong> {cameraInference.risk_score.toFixed(2)}</p>

                  {cameraInference.alert_generated && (
                    <div style={{ marginTop: '12px', padding: '10px', background: '#c0392b', borderRadius: '4px', border: '1px solid #e74c3c' }}>
                      <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: '11px' }}>{cameraInference.alert_generated}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: SPATIAL-TEMPORAL CONVLSTM */}
          {activeTab === 'temporal' && (
            <div>
              <h3 style={{ margin: '0 0 12px 0', color: '#f5f6fa', borderBottom: '1px solid #485460', paddingBottom: '8px' }}>
                Spatial-Temporal ConvLSTM Analysis
              </h3>
              <p style={{ fontSize: '12px', color: '#a4b0be' }}>
                Multi-spectral 10-day temporal sequences from 27-band MODIS/VIIRS NetCDF cubes predicting wildfire growth velocity and spread.
              </p>

              {temporalData && (
                <div style={{ fontSize: '13px', lineHeight: '1.6', background: '#2d3436', padding: '14px', borderRadius: '6px' }}>
                  <p style={{ margin: '0' }}><strong>Architecture:</strong> {temporalData.model}</p>
                  <p style={{ margin: '0' }}><strong>Dataset:</strong> {temporalData.dataset}</p>
                  <p style={{ margin: '0' }}><strong>Sequence Resolution:</strong> {temporalData.spatial_resolution} over {temporalData.temporal_sequence_length} days</p>
                  <p style={{ margin: '0' }}><strong>Predicted Spread Risk:</strong> <span style={{ color: '#ff4757', fontWeight: 'bold' }}>{temporalData.spread_risk}</span></p>
                  <p style={{ margin: '0' }}><strong>Estimated Growth Velocity:</strong> {temporalData.growth_velocity_km2_day} km²/day</p>
                  <p style={{ margin: '0' }}><strong>Model Confidence:</strong> {(temporalData.predicted_confidence * 100).toFixed(1)}%</p>

                  <h4 style={{ margin: '14px 0 6px 0', color: '#eccc68' }}>Temporal Attention Weights (Day 1 → Day 10)</h4>
                  <div style={{ display: 'flex', gap: '4px', alignItems: 'flex-end', height: '100px', background: '#1e272e', padding: '10px', borderRadius: '4px' }}>
                    {temporalData.temporal_attention_weights && temporalData.temporal_attention_weights.map((w, i) => (
                      <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <div style={{ width: '100%', height: `${Math.max(w * 350, 6)}px`, background: i >= 5 ? '#ff6b6b' : '#48dbfb', borderRadius: '2px' }} />
                        <span style={{ fontSize: '9px', marginTop: '4px' }}>D{i+1}</span>
                      </div>
                    ))}
                  </div>
                  <small style={{ color: '#747d8c', display: 'block', marginTop: '6px' }}>
                    *Higher attention weights on Days 6-10 indicate significant fire spread detection.
                  </small>
                </div>
              )}
            </div>
          )}

        </div>

        {/* Right Side GIS Map */}
        <div style={{ flex: 2, height: '680px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #2f3640' }}>
          <MapContainer 
            key={`${mapCenter[0]}-${mapCenter[1]}-${mapZoom}`} 
            center={mapCenter} 
            zoom={mapZoom} 
            style={{ height: '100%', width: '100%' }}>
            
            <TileLayer 
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" 
              attribution="&copy; OpenStreetMap contributors | NASA Earthdata FIRMS"
            />

            {/* Satellite Hotspots */}
            {activeTab === 'satellite' && nasaData.hotspots && nasaData.hotspots.map((spot, i) => (
              <Marker key={i} position={[spot.latitude, spot.longitude]}>
                <Popup>
                  <div style={{ fontSize: '12px' }}>
                    <strong style={{ color: '#d63031' }}>{spot.source}</strong><br/>
                    <strong>Fire Radiative Power:</strong> {spot.frp_mw.toFixed(1)} MW<br/>
                    <strong>Thermal Brightness:</strong> {spot.brightness_kelvin.toFixed(1)} K<br/>
                    <strong>Confidence:</strong> {spot.confidence}<br/>
                    <strong>Acquisition:</strong> {spot.acq_date} {spot.acq_time} UTC
                  </div>
                </Popup>
              </Marker>
            ))}

            {/* Camera Surveillance Ground Marker */}
            {activeTab === 'camera' && cameraInference && cameraInference.latitude && (
              <Marker position={[cameraInference.latitude, cameraInference.longitude]}>
                <Popup>
                  <div style={{ fontSize: '12px' }}>
                    <strong style={{ color: '#0984e3' }}>{cameraInference.source}</strong><br/>
                    <strong>Class:</strong> {cameraInference.predicted_class}<br/>
                    <strong>Severity:</strong> {cameraInference.severity}<br/>
                    <strong>Smoke Confidence:</strong> {(cameraInference.smoke_confidence * 100).toFixed(1)}%<br/>
                    <strong>Risk Score:</strong> {cameraInference.risk_score.toFixed(2)}
                  </div>
                </Popup>
              </Marker>
            )}

          </MapContainer>
        </div>

      </div>

    </div>
  );
}
