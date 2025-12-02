import React, { useState, useRef, useEffect, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { FaChevronLeft, FaPlay, FaPause, FaChartPie, FaMusic } from "react-icons/fa";
import { MdSpeed } from "react-icons/md";
import { useTheme } from "../context/ThemeContext";

const Result = ({ result, onBack, audioUrl, waveformLevels = [] }) => {
  const { t } = useTranslation();
  const { isDark } = useTheme();
  const [activeTab, setActiveTab] = useState("results"); // "audio" veya "results"
  const [isPlaying, setIsPlaying] = useState(false);
  const [playProgress, setPlayProgress] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  const audioRef = useRef(null);
  const speedMenuRef = useRef(null);
  const playbackAnimationRef = useRef(null);

  const speedOptions = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

  // Backend'den gelen veri yapısı: { prediction, confidence, all_probabilities, details }
  const { prediction, confidence, all_probabilities } = result;

  const translateEmotion = (emotion) => {
    const map = {
      angry: t('emotion_angry') || "Kızgın",
      calm: t('emotion_calm') || "Sakin",
      happy: t('emotion_happy') || "Mutlu",
      sad: t('emotion_sad') || "Üzgün"
    };
    return map[emotion] || emotion;
  };

  const getEmotionEmoji = (emotion) => {
    const map = { angry: "😡", calm: "😌", happy: "😊", sad: "😢" };
    return map[emotion] || "😐";
  };

  const getEmotionColor = (emotion) => {
    const map = {
      angry: "from-rose-500 to-orange-600",
      calm: "from-sky-400 to-teal-400",
      happy: "from-amber-400 to-orange-500",
      sad: "from-indigo-400 to-violet-600"
    };
    return map[emotion] || "from-gray-400 to-gray-500";
  };

  const getEmotionBg = (emotion) => {
    const map = {
      angry: "bg-rose-500",
      calm: "bg-sky-400",
      happy: "bg-amber-400",
      sad: "bg-indigo-400"
    };
    return map[emotion] || "bg-gray-400";
  };

  // Olasılıkları büyükten küçüğe sırala
  const sortedProbs = Object.entries(all_probabilities || {})
    .sort(([, a], [, b]) => b - a);

  const formatTime = (seconds) => {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(Math.floor(seconds % 60)).padStart(2, "0");
    return `${m}:${s}`;
  };

  const togglePlayPause = () => {
    if (!audioRef.current || !audioUrl) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    } else {
      if (audioRef.current.ended || audioRef.current.currentTime === 0) {
        audioRef.current.currentTime = 0;
      }
      audioRef.current.play();
      audioRef.current.playbackRate = playbackRate;
      setIsPlaying(true);
    }
  };

  const handleSpeedChange = (rate) => {
    setPlaybackRate(rate);
    if (audioRef.current) audioRef.current.playbackRate = rate;
    setIsSpeedMenuOpen(false);
  };

  const updatePlayProgressSmoothly = useCallback(() => {
    if (audioRef.current && !audioRef.current.paused && !audioRef.current.ended) {
      const duration = audioRef.current.duration;
      const currentTime = audioRef.current.currentTime;
      if (duration > 0) {
        setPlayProgress(currentTime / duration);
        setRecordingTime(currentTime);
      }
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgressSmoothly);
    }
  }, []);

  useEffect(() => {
    if (isPlaying) {
      playbackAnimationRef.current = requestAnimationFrame(updatePlayProgressSmoothly);
    } else if (playbackAnimationRef.current) {
      cancelAnimationFrame(playbackAnimationRef.current);
    }
    return () => {
      if (playbackAnimationRef.current) cancelAnimationFrame(playbackAnimationRef.current);
    };
  }, [isPlaying, updatePlayProgressSmoothly]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(event.target)) {
        setIsSpeedMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    if (audioRef.current && audioUrl) {
      const duration = audioRef.current.duration;
      if (!isNaN(duration)) {
        setRecordingTime(duration);
      }
    }
  }, [audioUrl]);

  return (
    <div className="w-full h-full flex flex-col animate-fadeIn p-4 md:p-6">
      {/* Üst Bar & Navigasyon */}
      <div className="flex items-center justify-between mb-8 flex-shrink-0 relative z-20">
        <div className="w-12">
          <button
            onClick={onBack}
            className={`group flex items-center justify-center w-12 h-12 rounded-full transition-all duration-300 backdrop-blur-md border ${isDark
                ? "bg-white/5 border-white/10 hover:bg-white/10 text-white"
                : "bg-white/60 border-gray-200 hover:bg-white text-gray-700"
              } shadow-lg hover:shadow-xl hover:-translate-x-1`}
          >
            <FaChevronLeft className="text-lg" />
          </button>
        </div>

        {/* Modern Segmented Control */}
        <div className="flex gap-4 mx-auto">
          <button
            onClick={() => setActiveTab("results")}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-md text-sm font-bold transition-all duration-300 border ${activeTab === "results"
                ? isDark
                  ? "bg-white/10 text-white shadow-lg border-white/10"
                  : "bg-white text-gray-900 shadow-md border-transparent"
                : isDark
                  ? "text-gray-400 hover:text-white border-transparent hover:bg-white/5"
                  : "text-gray-500 hover:text-gray-900 border-transparent hover:bg-gray-100"
              }`}
          >
            <FaChartPie className={activeTab === "results" ? "text-indigo-400" : ""} />
            {t('results') || "Analiz"}
          </button>
          <button
            onClick={() => setActiveTab("audio")}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-md text-sm font-bold transition-all duration-300 border ${activeTab === "audio"
                ? isDark
                  ? "bg-white/10 text-white shadow-lg border-white/10"
                  : "bg-white text-gray-900 shadow-md border-transparent"
                : isDark
                  ? "text-gray-400 hover:text-white border-transparent hover:bg-white/5"
                  : "text-gray-500 hover:text-gray-900 border-transparent hover:bg-gray-100"
              }`}
          >
            <FaMusic className={activeTab === "audio" ? "text-indigo-400" : ""} />
            {t('audio') || "Kayıt"}
          </button>
        </div>

        <div className="w-12" /> {/* Dengeleyici */}
      </div>

      <div className="flex-1 overflow-y-auto overflow-x-hidden custom-scrollbar w-full">
        <div className="min-h-full flex flex-col items-center justify-start w-full">

          {/* --- SONUÇLAR SEKMESİ --- */}
          {activeTab === "results" && (
            <div className="w-full max-w-4xl animate-slideUpFade pb-8 flex flex-col gap-8">
              {/* Ana Kart */}
              <div className={`relative overflow-hidden rounded-[2.5rem] p-8 md:p-12 text-center shadow-2xl transition-all duration-500 border ${isDark
                  ? "bg-gradient-to-br from-slate-800/80 to-slate-900/90 border-white/10"
                  : "bg-gradient-to-br from-white/90 to-indigo-50/90 border-white/60"
                } backdrop-blur-xl group`}>

                {/* Arka plan glow efektleri */}
                <div className={`absolute -top-32 -right-32 w-80 h-80 rounded-full opacity-20 blur-[100px] transition-colors duration-700 ${getEmotionBg(prediction)}`} />
                <div className={`absolute -bottom-32 -left-32 w-80 h-80 rounded-full opacity-20 blur-[100px] transition-colors duration-700 ${getEmotionBg(prediction)}`} />

                <div className="relative z-10 flex flex-col items-center">
                  {/* Emoji Container */}
                  <div className="relative mb-8">
                    <div className={`w-40 h-40 rounded-full flex items-center justify-center text-8xl shadow-2xl bg-gradient-to-br ${getEmotionColor(prediction)} animate-float border-4 border-white/20`}>
                      {getEmotionEmoji(prediction)}
                    </div>
                    {/* Ripple effect */}
                    <div className={`absolute inset-0 rounded-full bg-gradient-to-br ${getEmotionColor(prediction)} -z-10 animate-ping opacity-20`} />
                  </div>

                  {/* Duygu Başlığı */}
                  <div className="space-y-6 mb-8">
                    <h2 className={`text-lg font-medium tracking-widest uppercase opacity-60 ${isDark ? "text-white" : "text-gray-600"}`}>
                      {t('detected_emotion') || "Algılanan Duygu"}
                    </h2>
                    <h1 className={`text-5xl md:text-7xl font-black bg-gradient-to-r ${getEmotionColor(prediction)} bg-clip-text text-transparent drop-shadow-sm pb-4 leading-normal`}>
                      {translateEmotion(prediction)}
                    </h1>
                  </div>

                  {/* Güven Skoru Barı */}
                  <div className={`w-full max-w-lg p-6 rounded-3xl border backdrop-blur-md ${isDark ? "bg-white/5 border-white/5" : "bg-white/50 border-white/40"
                    }`}>
                    <div className="flex justify-between items-end mb-4">
                      <span className={`text-sm font-bold uppercase tracking-wider ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                        {t('confidence_score') || "Yapay Zeka Güveni"}
                      </span>
                      <span className={`text-3xl font-black ${isDark ? "text-white" : "text-gray-800"}`}>
                        %{Math.round(confidence * 100)}
                      </span>
                    </div>
                    <div className={`h-4 rounded-full overflow-hidden p-1 ${isDark ? "bg-black/40" : "bg-gray-200"}`}>
                      <div
                        className={`h-full rounded-full bg-gradient-to-r ${getEmotionColor(prediction)} shadow-lg relative overflow-hidden`}
                        style={{ width: `${confidence * 100}%`, transition: "width 1.5s cubic-bezier(0.34, 1.56, 0.64, 1)" }}
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent animate-shimmer"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Detaylı İstatistikler Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {sortedProbs.map(([emotion, score], index) => (
                  <div
                    key={emotion}
                    className={`p-5 rounded-2xl border flex items-center gap-4 transition-all duration-300 hover:scale-[1.02] ${isDark
                        ? "bg-slate-800/40 border-white/5 hover:bg-slate-800/60"
                        : "bg-white/60 border-white/40 hover:bg-white/80"
                      } backdrop-blur-md shadow-sm`}
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl bg-gradient-to-br ${getEmotionColor(emotion)} shadow-lg text-white`}>
                      {getEmotionEmoji(emotion)}
                    </div>

                    <div className="flex-1">
                      <div className="flex justify-between mb-2">
                        <span className={`font-bold text-lg capitalize ${isDark ? "text-gray-200" : "text-gray-700"}`}>
                          {translateEmotion(emotion)}
                        </span>
                        <span className={`font-mono font-medium ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                          %{(score * 100).toFixed(1)}
                        </span>
                      </div>
                      <div className={`h-2.5 rounded-full overflow-hidden ${isDark ? "bg-black/20" : "bg-gray-200/80"}`}>
                        <div
                          className={`h-full rounded-full bg-gradient-to-r ${getEmotionColor(emotion)} opacity-90`}
                          style={{ width: `${score * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* --- SES SEKMESİ --- */}
          {activeTab === "audio" && audioUrl && (
            <div className="w-full max-w-3xl animate-slideUpFade pb-8">

              {/* Görselleştirici Kartı */}
              <div className={`w-full rounded-[2.5rem] p-8 mb-8 border relative overflow-hidden ${isDark
                  ? "bg-slate-800/60 border-white/5"
                  : "bg-white/60 border-white/40"
                } backdrop-blur-xl shadow-xl`}>

                {/* Waveform */}
                <div className="h-48 flex items-center justify-center gap-1.5 w-full mb-8">
                  {waveformLevels.map((level, index) => {
                    const totalBars = waveformLevels.length;
                    const playIndex = Math.floor(playProgress * (totalBars - 1));
                    const isCurrent = index === playIndex && isPlaying;
                    const isPassed = index < playIndex;

                    return (
                      <div
                        key={index}
                        className={`flex-1 rounded-full transition-all duration-150 ease-out
                        ${isCurrent
                            ? 'bg-indigo-500 shadow-[0_0_20px_rgba(99,102,241,0.6)] scale-110'
                            : isPassed
                              ? isDark ? 'bg-indigo-400/60' : 'bg-indigo-400'
                              : isDark ? 'bg-slate-700' : 'bg-gray-300'
                          }`}
                        style={{
                          height: `${Math.max(8, level * 160)}px`,
                          opacity: isCurrent ? 1 : isPassed ? 0.8 : 0.4,
                        }}
                      />
                    );
                  })}
                </div>

                {/* Kontroller */}
                <div className="flex flex-col items-center gap-8">
                  <div className={`font-mono text-5xl font-black tracking-widest ${isDark ? "text-white" : "text-gray-800"
                    }`}>
                    {formatTime(recordingTime)}
                  </div>

                  <div className="flex items-center gap-8">
                    {/* Hız Butonu */}
                    <div className="relative" ref={speedMenuRef}>
                      <button
                        onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
                        className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-300 ${isDark
                            ? "bg-white/5 hover:bg-white/10 text-white border border-white/10"
                            : "bg-white hover:bg-gray-50 text-gray-700 border border-gray-200 shadow-sm"
                          }`}
                      >
                        {playbackRate === 1.0 ? (
                          <MdSpeed className="text-2xl" />
                        ) : (
                          <span className="text-sm font-bold">{playbackRate}x</span>
                        )}
                      </button>

                      {isSpeedMenuOpen && (
                        <div className={`absolute bottom-full mb-4 left-1/2 -translate-x-1/2 w-32 rounded-2xl shadow-2xl overflow-hidden py-2 z-[100] ${isDark ? "bg-slate-800 border border-white/10" : "bg-white border border-gray-100"
                          } animate-fadeIn`}>
                          {speedOptions.map((rate) => (
                            <button
                              key={rate}
                              onClick={() => handleSpeedChange(rate)}
                              className={`block w-full text-center px-4 py-2 text-sm font-bold transition-colors
                                    ${rate === playbackRate
                                  ? 'bg-indigo-500 text-white'
                                  : isDark
                                    ? 'text-gray-300 hover:bg-white/5'
                                    : 'text-gray-600 hover:bg-gray-50'
                                }`}
                            >
                              {rate}x
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Play/Pause */}
                    <button
                      onClick={togglePlayPause}
                      className="w-24 h-24 rounded-3xl bg-gradient-to-br from-indigo-500 to-violet-600 hover:from-indigo-400 hover:to-violet-500 flex items-center justify-center text-white shadow-2xl shadow-indigo-500/30 hover:scale-105 hover:shadow-indigo-500/50 transition-all duration-300 group"
                    >
                      {isPlaying ? (
                        <FaPause className="text-3xl" />
                      ) : (
                        <FaPlay className="text-3xl ml-2 group-hover:scale-110 transition-transform" />
                      )}
                    </button>

                    {/* Boşluk (Simetri için) */}
                    <div className="w-14 h-14 opacity-0" />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Result;