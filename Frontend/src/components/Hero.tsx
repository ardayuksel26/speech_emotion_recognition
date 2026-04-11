import { useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import AudioInput from "./AudioInput/AudioInput";
import AudioPlayer from "./AudioPlayer";
import Result from "./Result";
import axios from "axios";
import { convertFileToWav } from "../utils/audioUtils";
import { AnalysisResult } from "../types";
import InteractiveBackground from "./InteractiveBackground";
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

      const { final_emotion, confidence, main_scores, veto_applied, rf_sad_score } = response.data;

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
         
         // Normalize other emotions to fill the remaining space exactly, without modifying Sad.
         const otherSum = Object.values(emotionsMap).reduce((a, b) => a + b, 0) - rfScoreNum;
         if (otherSum > 0) {
             const scale = (1.0 - rfScoreNum) / otherSum;
             Object.keys(emotionsMap).forEach(key => {
                 if (key !== 'sad') {
                     emotionsMap[key] = emotionsMap[key] * scale;
                 }
             });
         }
      } else {
         const totalSum = Object.values(emotionsMap).reduce((a, b) => a + b, 0);
         if (totalSum > 0) {
           Object.keys(emotionsMap).forEach(key => {
             emotionsMap[key] = emotionsMap[key] / totalSum;
           });
         }
      }

      // Format custom Mastermind details for the frontend
      const r_sad_value = parseFloat(String(rf_sad_score).replace('%', ''));

      setAnalysisResult({
        dominant_emotion: final_emotion,
        emotions: emotionsMap,
        confidence: normalizedConfidence,
        word_timestamps: response.data.word_timestamps || [],
        veto_info: {
            applied: veto_applied,
            rf_score: r_sad_value
        }
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
    <div className={`relative w-full flex-grow flex flex-col items-center justify-center overflow-hidden font-sans transition-colors duration-500 ${isDark ? "bg-[#0b0f19] text-white" : "bg-gray-50 text-slate-900"}`}>
      
      {/* Interactive Background Canvas */}
      <InteractiveBackground />

      <div className="relative z-10 w-full max-w-6xl px-6 flex flex-col items-center py-24 mb-10">

        <h1 className={`font-outfit text-6xl md:text-8xl font-black mb-6 py-2 leading-tight text-center tracking-tighter transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] ${analysisResult ? "scale-75 mb-0 opacity-0 h-0" : "opacity-100"}`}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-fuchsia-500 to-rose-500 drop-shadow-sm">
            {t('discover_your_voice')}
          </span>
        </h1>
        
        {!analysisResult && (
           <p className={`text-xl md:text-2xl font-light mb-16 text-center max-w-3xl leading-relaxed tracking-wide ${isDark ? "text-indigo-100/70" : "text-slate-600/90"} animate-slideUpFade`} style={{ animationDelay: '0.1s' }}>
              Gelişmiş <strong className="font-semibold text-indigo-400 dark:text-indigo-300">Üst Akıl</strong> algoritması ile sesinizdeki 4 temel duyguyu anında ve kusursuzca analiz edin. Sadece konuşun, gerisini biz halledelim.
           </p>
        )}

        {/* Main Interface Module */}
        <div className={`
          relative w-full backdrop-blur-[40px] shadow-2xl transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)]
          flex flex-col items-center justify-center border
          ${isDark ? "bg-[#0f172a]/70 border-white/10 shadow-[0_0_100px_rgba(99,102,241,0.15)]" : "bg-white/70 border-indigo-100/80 shadow-[0_0_100px_rgba(99,102,241,0.1)]"}
          ${analysisResult ? "max-w-[98vw] lg:max-w-[1600px] min-h-[85vh] p-2 md:p-6 overflow-hidden rounded-[2.5rem] mx-auto border-indigo-500/20" : "max-w-4xl min-h-[500px] p-8 md:p-14 rounded-[3rem]"}
        `}>
          
          {/* Subtle Inner Glow */}
          {!analysisResult && (
             <div className="absolute inset-0 rounded-[3rem] pointer-events-none shadow-[inset_0_0_60px_rgba(255,255,255,0.05)]" />
          )}

          {!audioFile && (
            <div className="w-full flex-1 flex flex-col items-center justify-center animate-fadeIn z-10 relative">
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
            <div className="w-full h-full p-2 md:p-4 overflow-y-auto animate-fadeIn custom-scrollbar">
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
