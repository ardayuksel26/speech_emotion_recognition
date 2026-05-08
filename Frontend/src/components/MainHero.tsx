import { useState } from "react";
import { useTranslation, Trans } from "react-i18next";
import { useTheme } from "../context/ThemeContext";
import { clsx } from "clsx";
import AudioInput from "./AudioInput/AudioInput";
import Result from "./Result";
import axios from "axios";
import { convertFileToWav } from "../utils/audioUtils";
import { AnalysisResult } from "../types";





const MainHero = () => {
  const { t } = useTranslation();
  const { isDark } = useTheme();

  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);

  /* ── Ses hazır → Vosk segmentasyon + otomatik analiz ── */
  const handleAudioReady = async (file: File) => {
    const url = URL.createObjectURL(file);
    setRecordedUrl(url);
    await runSegmentAndAnalyze(file);
  };

  const runSegmentAndAnalyze = async (file: File) => {
    setIsSegmenting(true);
    setAnalysisResult(null);

    try {
      const wavBlob = await convertFileToWav(file);
      const formData = new FormData();
      formData.append("file", wavBlob, "converted_audio.wav");

      // Master Ensemble: V2 (cb_v2 + lgbm_v2 + xgb_v2) kelime bazlı +
      // HuBERT cümle jürisi → F1-ağırlıklı birleştirme
      setIsSegmenting(false);
      setIsAnalyzing(true);

      const response = await axios.post(`${import.meta.env.VITE_API_URL}/analyze_master`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { emotion, confidence, all_scores, model_details, word_timestamps } = response.data;

      let confidenceValue = 0;
      if (typeof confidence === "string") confidenceValue = parseFloat(confidence.replace("%", ""));
      else if (typeof confidence === "number") confidenceValue = confidence;
      const normalizedConfidence = confidenceValue / 100;

      const emotionsMap: { [key: string]: number; happy: number; sad: number; angry: number; calm: number } = {
        happy: 0, sad: 0, angry: 0, calm: 0,
      };
      if (all_scores) {
        Object.entries(all_scores).forEach(([key, val]) => {
          emotionsMap[key.toLowerCase()] = (val as number) / 100;
        });
      } else {
        emotionsMap[emotion.toLowerCase()] = normalizedConfidence;
      }
      const totalSum = Object.values(emotionsMap).reduce((a, b) => a + b, 0);
      if (totalSum > 0) Object.keys(emotionsMap).forEach(k => { emotionsMap[k] = emotionsMap[k] / totalSum; });

      // Her kelime için V2 modellerinin tahminlediği duyguyu timeline'a ver
      const finalWordTimestamps = (word_timestamps && word_timestamps.length > 0)
        ? word_timestamps.map((w: any) => ({
            word: w.word || "—",
            start: w.start,
            end: w.end,
            emotion: w.emotion || emotion.toLowerCase(),
            confidence: typeof w.confidence === "number" ? w.confidence : normalizedConfidence,
          }))
        : [];

      setAnalysisResult({
        dominant_emotion: emotion,
        emotions: emotionsMap,
        confidence: normalizedConfidence,
        word_timestamps: finalWordTimestamps,
        model_details: model_details || undefined,
      });
    } catch (error: any) {
      console.error("Master analysis failed:", error);
      let msg = t("analysis_failed") || "Analiz hatası.";
      if (error.response?.data?.error) msg = `Hata: ${error.response.data.error}`;
      else if (error.message) msg = `Hata: ${error.message}`;
      alert(msg);
    } finally {
      setIsSegmenting(false);
      setIsAnalyzing(false);
    }
  };


  const reset = () => {
    setRecordedUrl(null);
    setAnalysisResult(null);
    setIsSegmenting(false);
    setIsAnalyzing(false);
  };

  return (
    <div
      className={clsx(
        "relative w-full flex-grow flex flex-col items-center font-sans",
        analysisResult ? "justify-start overflow-x-hidden" : "justify-center"
      )}
      style={analysisResult ? { paddingTop: '80px', paddingBottom: '24px' } : { paddingTop: '32px', paddingBottom: '32px' }}
    >


      <div className={clsx(
        "relative z-10 w-full max-w-6xl px-4 sm:px-6 flex flex-col items-center",
        analysisResult ? "" : "py-4 sm:py-8 md:py-12"
      )}
      style={analysisResult ? { paddingBottom: '4px' } : { paddingBottom: '40px' }}
      >

        {/* Başlık */}
        <h1 className={clsx(
          "font-outfit text-3xl sm:text-5xl md:text-7xl font-black leading-tight text-center tracking-tighter transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]",
          analysisResult 
            ? "scale-75 mb-0 opacity-0 h-0 overflow-hidden py-0" 
            : "opacity-100 mb-4 sm:mb-6 py-1 sm:py-2"
        )}>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-fuchsia-500 to-rose-500 drop-shadow-sm">
            {t("discover_your_voice")}
          </span>
        </h1>

        {!analysisResult && (
          <p className={`text-base sm:text-xl md:text-2xl font-light mb-8 sm:mb-12 md:mb-16 text-center max-w-3xl leading-relaxed tracking-wide ${isDark ? "text-indigo-100/70" : "text-slate-600/90"} animate-slideUpFade`} style={{ animationDelay: "0.1s" }}>
            <Trans i18nKey="hero_subtitle">
              Gelişmiş <strong className="font-semibold text-indigo-400 dark:text-indigo-300">Üst Akıl</strong> algoritması ile sesinizdeki 4 temel duyguyu anında ve kusursuzca analiz edin.
            </Trans>
          </p>
        )}

        {/* Ana Kart */}
        <div className={clsx(
          "relative w-full shadow-2xl transition-all duration-1000 ease-[cubic-bezier(0.16,1,0.3,1)] mt-8 sm:mt-0",
          "flex flex-col items-center justify-center border",
          isDark ? "bg-[#0f172a]/70 backdrop-blur-[40px] border-white/10 shadow-[0_0_100px_rgba(99,102,241,0.15)]" : "bg-white/60 backdrop-blur-[40px] border-indigo-100 shadow-[0_0_60px_rgba(99,102,241,0.08)]",
          analysisResult
            ? "max-w-[100vw] sm:max-w-[98vw] lg:max-w-[1600px] overflow-visible rounded-3xl md:rounded-[2.5rem] mx-auto border-indigo-500/20"
            : "max-w-5xl min-h-[320px] px-4 py-8 sm:p-6 md:p-10 rounded-[3rem] w-full mx-4 md:mx-0"
        )}
          style={!analysisResult ? { minHeight: '280px' } : { marginTop: '0', padding: '0 8px 4px' }}
        >
          {!analysisResult && (
            <div className="absolute inset-0 rounded-[3rem] pointer-events-none shadow-[inset_0_0_60px_rgba(255,255,255,0.05)]" />
          )}

          {/* Faz 1: Ses yok */}
          {!isSegmenting && !isAnalyzing && !analysisResult && (
            <div className="w-full flex-1 flex flex-col items-center justify-center animate-fadeIn z-10 relative">
              <AudioInput onAudioReady={handleAudioReady} compact />
            </div>
          )}

          {/* Faz 2: Segmentasyon spinner */}
          {isSegmenting && (
            <div className="flex flex-col items-center justify-center w-full h-full animate-fadeIn py-20 gap-4">
              <div className="relative w-20 h-20">
                <div className="absolute inset-0 border-4 border-indigo-200/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-indigo-600 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                <div className="absolute inset-2 border-4 border-purple-600 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse">
                {t("segmenting")}
              </p>
            </div>
          )}

          {/* Faz 3: Analiz spinner */}
          {isAnalyzing && (
            <div className="flex flex-col items-center justify-center w-full h-full animate-fadeIn py-20">
              <div className="relative w-24 h-24 mb-6">
                <div className="absolute inset-0 border-4 border-indigo-200/20 rounded-full" />
                <div className="absolute inset-0 border-4 border-indigo-600 border-t-transparent rounded-full animate-[spin_1s_linear_infinite]" />
                <div className="absolute inset-2 border-4 border-purple-600 border-b-transparent rounded-full animate-[spin_1.5s_linear_infinite_reverse]" />
              </div>
              <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500 animate-pulse">
                {t("analyzing")}
              </p>
            </div>
          )}

          {/* Faz 4: Sonuç */}
          {analysisResult && (
            <div className="w-full animate-fadeIn" style={{ height: '100%' }}>
              <Result result={analysisResult} onBack={reset} audioUrl={recordedUrl || undefined} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MainHero;
