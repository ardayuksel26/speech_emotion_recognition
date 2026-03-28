import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import AudioInput from "./AudioInput/AudioInput";
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

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [savedLevels, setSavedLevels] = useState<number[]>([]);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

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
      formData.append("audio", wavBlob, "mastermind_audio.wav");

      // No settings, no hassle. Send directly to Mastermind.
      const response = await axios.post(`http://localhost:5000/api/predict_mastermind`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { final_emotion, confidence, main_scores, veto_applied, rf_sad_score, segments_analyzed } = response.data;

      // Parse confidence string (e.g., "%95.20" -> 95.20)
      let confidenceValue = 0;
      if (typeof confidence === 'string') {
        confidenceValue = parseFloat(confidence.replace('%', ''));
      } else if (typeof confidence === 'number') {
        confidenceValue = confidence;
      }

      const normalizedConfidence = confidenceValue / 100;

      // Map backend Mastermind scores to Frontend standard 0-1 scale
      const emotionsMap: { [key: string]: number; happy: number; sad: number; angry: number; calm: number; } = {
        happy: 0,
        sad: 0,
        angry: 0,
        calm: 0
      };
      
      if (main_scores) {
        Object.entries(main_scores).forEach(([key, val]) => {
          emotionsMap[key.toLowerCase()] = (val as number) / 100;
        });
      }

      // If veto applied, we forcefully set the primary emotion score in graph to reflect reality
      if (veto_applied) {
         const rfScoreNum = parseFloat(String(rf_sad_score).replace('%', '')) / 100;
         emotionsMap['sad'] = rfScoreNum;
      }

      const totalSum = Object.values(emotionsMap).reduce((a, b) => a + b, 0);
      if (totalSum > 0) {
        Object.keys(emotionsMap).forEach(key => {
          emotionsMap[key] = emotionsMap[key] / totalSum;
        });
      }

      // Format custom Mastermind details for the frontend
      const mastermindDetails = {
        "AI Modeli": "The Mastermind (Ensemble)",
        "Segmentasyon": "VOSK Engine",
        "İncelenen Kelime": `${segments_analyzed} adet`,
        "VETO Durumu": veto_applied ? `Aktif (%${parseFloat(String(rf_sad_score).replace('%', '')).toFixed(1)} Sad)` : "Pasif"
      };

      setAnalysisResult({
        dominant_emotion: final_emotion,
        emotions: emotionsMap,
        confidence: normalizedConfidence,
        word_timestamps: [], // Clean view
        model_details: mastermindDetails
      });

      setIsAnalyzing(false);

    } catch (error: any) {
      console.error("Mastermind Request Failed:", error);
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

        <h1 className={`text-5xl md:text-7xl font-extrabold mb-4 py-2 leading-relaxed text-center tracking-tight transition-all duration-500 ${isDark ? "text-white" : "text-gray-900"} ${analysisResult ? "scale-75 mb-0" : ""}`}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
            {t('discover_your_voice')}
          </span>
        </h1>
        
        {!analysisResult && (
           <p className={`text-lg md:text-xl font-medium mb-12 text-center max-w-2xl ${isDark ? "text-slate-400" : "text-slate-600"}`}>
              Gelişmiş "Üst Akıl" algoritması ile sesinizdeki 4 temel duyguyu anında ve kusursuzca analiz edin. Sadece konuşun, gerisini biz halledelim.
           </p>
        )}

        <div className={`
          relative w-full backdrop-blur-2xl rounded-3xl shadow-2xl transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]
          flex flex-col items-center justify-center
          ${isDark ? "bg-slate-800/60 border border-white/5" : "bg-white/90 border border-slate-200/50"}
          ${analysisResult ? "max-w-[95vw] lg:max-w-[1400px] min-h-[85vh] p-0 overflow-hidden" : "max-w-4xl min-h-[450px] p-8 md:p-12"}
        `}>

          {!audioFile && (
            <div className="w-full flex-1 flex flex-col items-center justify-center animate-fadeIn">
              <AudioInput onAudioReady={handleAudioReady} />
            </div>
          )}

          {audioFile && recordedUrl && !analysisResult && !isAnalyzing && (
            <div className="w-full max-w-2xl flex flex-col items-center animate-fadeIn">
              
              <div className="w-full mb-8 p-1">
                 <AudioPlayer
                   mode="preview"
                   analysisMode="word"
                   selectedModelName="The Mastermind AI"
                   recordedUrl={recordedUrl}
                   showAnalyzeButton={true}
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
                   isRecording={false}
                   recordingTime={currentTime > 0 ? currentTime : duration}
                   duration={duration}
                   currentTime={currentTime}
                   onStartRecording={() => { }}
                   onStopRecording={() => { }}
                 />
              </div>
            </div>
          )}

          {isAnalyzing && (
            <div className="flex flex-col items-center justify-center w-full h-full animate-fadeIn py-20">
              <div className="relative w-24 h-24 mb-6">
                 <div className="absolute inset-0 border-4 border-indigo-200/20 dark:border-indigo-500/20 rounded-full" />
                 <div className="absolute inset-0 border-4 border-indigo-600 dark:border-indigo-400 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                 <div className="absolute inset-2 border-4 border-purple-600 dark:border-purple-400 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse">
                Üst Akıl Analiz Ediyor...
              </p>
              <p className="text-sm font-medium text-slate-500 dark:text-slate-400 mt-3 tracking-widest uppercase">
                VOSK 🚀 MULTI-MODEL SYNTHESIS
              </p>
            </div>
          )}

          {analysisResult && (
            <div className="w-full h-full p-6 md:p-8 overflow-y-auto animate-fadeIn">
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
