import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import AudioInput from "./AudioInput/AudioInput"; // Using the new component
import AudioPlayer from "./AudioPlayer";
import Result from "./Result";
import axios from "axios";
import { convertFileToWav } from "../utils/audioUtils";

import { AnalysisResult } from "../types";

// Helper to generate levels for visualization from a file
const generateWaveLevels = async (file: File, bars: number): Promise<number[]> => {
  const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
  const ctx = new AudioCtx();
  const ab = await file.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(ab);
  const rawData = audioBuffer.getChannelData(0);
  const blockSize = Math.floor(rawData.length / bars);
  const samples = [];
  for (let i = 0; i < bars; i++) {
    let sum = 0;
    for (let j = 0; j < blockSize; j++) {
      sum += Math.abs(rawData[i * blockSize + j]);
    }
    samples.push(sum / blockSize);
  }
  const max = Math.max(...samples, 0.01);
  return samples.map(s => s / max);
};

const Hero = () => {
  const { t } = useTranslation();
  const { isDark } = useTheme();

  // Available models list
  const MODELS = [
    { id: 'catboost', name: 'CatBoost' },
    { id: 'xgboost', name: 'XGBoost' },
    { id: 'lightgbm', name: 'LightGBM' },
    { id: 'rf', name: 'Random Forest' },
    { id: 'knn', name: 'K-Nearest Neighbors (KNN)' },
    { id: 'svm', name: 'Support Vector Machine (SVM)' },
    { id: 'mlp', name: 'Multi-Layer Perceptron (MLP)' },
    { id: 'gradient_boosting', name: 'Gradient Boosting' },
    { id: 'dnn', name: 'Deep Neural Network (DNN)' },
    { id: 'cnn1d', name: '1D Convolutional Neural Network' },
  ];

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [savedLevels, setSavedLevels] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedModel, setSelectedModel] = useState('catboost');
  const [mode, setMode] = useState<'word' | 'sentence'>('word');

  // Audio Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);

  const audioElementRef = useRef<HTMLAudioElement>(null);
  const playbackAnimationRef = useRef<number | null>(null);

  const handleAudioReady = async (file: File) => {
    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setRecordedUrl(url);

    // Generate levels for preview
    const levels = await generateWaveLevels(file, 60);
    setSavedLevels(levels);
  };

  const tooglePlayPause = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    } else {
      audioElementRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleSpeedChange = (rate: number) => {
    setPlaybackRate(rate);
    if (audioElementRef.current) audioElementRef.current.playbackRate = rate;
  };

  const updatePlayProgress = () => {
    if (audioElementRef.current) {
      setPlayProgress(audioElementRef.current.currentTime / audioElementRef.current.duration);
      setCurrentTime(audioElementRef.current.currentTime);
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgress);
    }
  };

  useEffect(() => {
    if (isPlaying) {
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgress);
    } else {
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    }
    return () => {
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    };
  }, [isPlaying]);

  const handleAnalyze = async () => {
    if (!audioFile) return;
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const wavBlob = await convertFileToWav(audioFile);
      const formData = new FormData();
      formData.append("file", wavBlob, "converted_audio.wav"); // Send converted WAV
      formData.append("model_type", selectedModel);

      // Make API Request
      // Make API Request
      // Using the new flask backend at port 5000
      const endpoint = mode === 'word' ? '/predict' : '/predict-sentence';
      const response = await axios.post(`http://localhost:5000${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      // Special handling for sentence mode response
      let emotion, confidence, all_scores;
      // let breakdown, weighted_details; // Placeholder for future use using breakdown and weighted details

      if (mode === 'sentence') {
        // Sentence endpoint returns different structure
        emotion = response.data.final_emotion;
        confidence = response.data.confidence;
        // Use weighted details if available, otherwise fallback
        if (response.data.weighted_details) {
          all_scores = response.data.weighted_details;
        }
        // breakdown = response.data.breakdown;
      } else {
        // Standard /predict response
        emotion = response.data.emotion;
        confidence = response.data.confidence;
        all_scores = response.data.all_scores;
      }

      // Parse confidence string (e.g., "%95.20" -> 95.20)
      let confidenceValue = 0;
      if (typeof confidence === 'string') {
        confidenceValue = parseFloat(confidence.replace('%', ''));
      } else if (typeof confidence === 'number') {
        confidenceValue = confidence;
      }

      // Backend returns 0-100 range, Frontend expects 0-1 range
      const normalizedConfidence = confidenceValue / 100;

      // Process all_scores if available (Backend returns 0-100)
      const emotionsMap: { [key: string]: number; happy: number; sad: number; angry: number; calm: number; } = {
        happy: 0,
        sad: 0,
        angry: 0,
        calm: 0
      };
      if (all_scores) {
        Object.entries(all_scores).forEach(([key, val]) => {
          emotionsMap[key.toLowerCase()] = (val as number) / 100;
        });
      } else {
        // Fallback for older backend response
        emotionsMap[emotion.toLowerCase()] = normalizedConfidence;
      }

      setAnalysisResult({
        dominant_emotion: emotion,
        emotions: emotionsMap,
        confidence: normalizedConfidence,
        // Backend doesn't support word timestamps yet
        word_timestamps: []
      });

      setIsAnalyzing(false);

    } catch (error: any) {
      console.error("Analysis Request Failed:", error);
      setIsAnalyzing(false);
      let msg = t('analysis_failed') || "Analiz hatası.";
      if (error.response?.data?.error) {
        msg = `Hata: ${error.response.data.error}`;
      } else if (error.message) {
        msg = `Hata: ${error.message}`;
      }
      alert(msg);
    }
  };

  const reset = () => {
    setAudioFile(null);
    setRecordedUrl(null);
    setAnalysisResult(null);
    setSavedLevels([]);
    setIsPlaying(false);
    setPlayProgress(0);
  };

  return (
    <div className={`relative w-full flex-grow flex flex-col items-center justify-center overflow-hidden font-sans transition-colors duration-300 ${isDark ? "bg-slate-900" : "bg-gray-100"}`}>
      <div className="relative z-10 w-full max-w-5xl px-6 flex flex-col items-center py-20">



        <h1 className={`text-4xl md:text-6xl font-extrabold mb-10 py-2 leading-relaxed text-center transition-all duration-500 ${isDark ? "text-white" : "text-gray-800"} ${analysisResult ? "scale-75 mb-4" : "mb-10"}`}>
          {t('discover_your_voice')}
        </h1>

        <div className={`
          relative w-full backdrop-blur-xl rounded-2xl shadow-2xl transition-all duration-500 ease-in-out
          flex flex-col items-center justify-center
          ${isDark ? "bg-slate-800/40 border border-white/10" : "bg-white/80 border border-gray-200"}
          ${analysisResult ? "max-w-[95vw] lg:max-w-[1400px] min-h-[85vh] p-0 overflow-hidden" : "max-w-5xl min-h-[400px] p-8"}
        `}>

          {/* Mode Switcher */}
          {!analysisResult && (
            <div className="flex bg-slate-200 dark:bg-slate-700 p-1 rounded-full mb-8 relative z-20">
              <button
                onClick={() => setMode('word')}
                className={`px-6 py-2 rounded-full text-sm font-bold transition-all duration-300 ${mode === 'word'
                  ? 'bg-white dark:bg-slate-600 text-indigo-600 dark:text-white shadow-md'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700'
                  }`}
              >
                Kelime (Word)
              </button>
              <button
                onClick={() => setMode('sentence')}
                className={`px-6 py-2 rounded-full text-sm font-bold transition-all duration-300 ${mode === 'sentence'
                  ? 'bg-white dark:bg-slate-600 text-indigo-600 dark:text-white shadow-md'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700'
                  }`}
              >
                Cümle (Sentence)
              </button>
            </div>
          )}

          {!audioFile && (
            <AudioInput onAudioReady={handleAudioReady} />
          )}

          {audioFile && recordedUrl && !analysisResult && !isAnalyzing && (
            <div className="w-full max-w-2xl flex flex-col items-center animate-fadeIn">

              {/* Model Selection UI */}
              <div className="w-full mb-8 p-6 bg-white dark:bg-slate-800/50 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
                <label className="block text-sm font-semibold text-slate-600 dark:text-slate-300 mb-3 text-center">
                  🧠 Analiz Modelini Seçin
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {MODELS.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModel(model.id)}
                      className={`
                        py-3 px-4 rounded-xl text-sm font-medium transition-all duration-200 border-2
                        ${selectedModel === model.id
                          ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300 shadow-md transform scale-105'
                          : 'border-transparent bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600'}
                      `}
                    >
                      {model.name}
                    </button>
                  ))}
                </div>
              </div>

              <AudioPlayer
                mode="preview"
                analysisMode={mode}
                recordedUrl={recordedUrl}
                levels={savedLevels}
                isPlaying={isPlaying}
                playProgress={playProgress}
                playbackRate={playbackRate}
                onTogglePlay={tooglePlayPause}
                onSpeedChange={handleSpeedChange}
                onAnalyze={handleAnalyze}
                onBack={reset}
                isSpeedMenuOpen={isSpeedMenuOpen}
                setIsSpeedMenuOpen={setIsSpeedMenuOpen}
                // Props not needed for simple preview but required by component signature:
                isRecording={false}
                recordingTime={currentTime > 0 ? currentTime : duration}
                duration={duration}
                currentTime={currentTime}
                onStartRecording={() => { }}
                onStopRecording={() => { }}
              />
            </div>
          )}

          {isAnalyzing && (
            <div className="flex flex-col items-center animate-pulse py-20">
              <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-xl font-medium text-indigo-500">{t('analyzing')}</p>
              <p className="text-sm text-slate-400 mt-2">Seçilen Model: {MODELS.find(m => m.id === selectedModel)?.name}</p>
            </div>
          )}

          {analysisResult && (
            <div className="w-full h-full p-6 md:p-8 overflow-y-auto">
              <Result
                result={analysisResult}
                onBack={reset}
                audioUrl={recordedUrl || undefined}
              />
            </div>
          )}

          {recordedUrl && (
            <audio
              ref={audioElementRef}
              src={recordedUrl}
              onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onEnded={() => { setIsPlaying(false); setPlayProgress(0); setCurrentTime(0); }}
              className="hidden"
            />
          )}

        </div>
      </div>
    </div>
  );
};

export default Hero;