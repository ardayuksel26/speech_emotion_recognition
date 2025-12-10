import { FaMicrophone, FaPlay, FaPause, FaMagic, FaChevronLeft } from "react-icons/fa";
import { MdSpeed } from "react-icons/md";
import { useRef, useEffect } from "react";
import { useTheme } from "../context/ThemeContext";
import { useTranslation } from "react-i18next";

const AudioPlayer = ({
  mode,
  isRecording,
  recordedUrl,
  recordingTime,
  levels,
  isPlaying,
  playProgress,
  playbackRate,
  onStartRecording,
  onStopRecording,
  onTogglePlay,
  onSpeedChange,
  onAnalyze,
  onBack,
  isSpeedMenuOpen,
  setIsSpeedMenuOpen,
  duration, // New prop
  currentTime, // New prop
}) => {
  const { isDark } = useTheme();
  const { t } = useTranslation();
  const speedMenuRef = useRef(null);
  const speedOptions = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];

  const formatTime = (seconds) => {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(Math.floor(seconds % 60)).padStart(2, "0");
    return `${m}:${s}`;
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (speedMenuRef.current && !speedMenuRef.current.contains(event.target)) {
        setIsSpeedMenuOpen(false); // Prop olan fonksiyonu çağırır
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [setIsSpeedMenuOpen]); // Dependency array güncellendi

  return (
    <div className="w-full flex flex-col items-center animate-fade-in-up relative">

      {/* Geri Butonu - Sadece kayıt yapıldıysa göster */}
      {recordedUrl && !isRecording && (
        <button
          onClick={onBack}
          className={`absolute -top-6 left-0 p-3 rounded-full transition-colors flex items-center justify-center z-30 ${isDark ? "hover:bg-white/10 text-gray-300" : "hover:bg-gray-100 text-gray-600"
            }`}
        >
          <FaChevronLeft className="text-lg" />
        </button>
      )}

      <div className={`w-full h-40 rounded-2xl border p-4 mb-4 relative overflow-hidden flex items-center justify-center ${isDark
        ? "bg-slate-800/60 border-white/5"
        : "bg-purple-100 border-purple-200"
        }`}>
        {!isRecording && !recordedUrl ? (
          <div className={`flex flex-col items-center justify-center h-full animate-pulse ${isDark ? "text-slate-500" : "text-purple-600"
            }`}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${isDark ? "bg-white/5" : "bg-purple-200"
              }`}>
              <FaMicrophone className={`text-lg ${isDark ? "text-slate-400" : "text-purple-600"
                }`} />
            </div>
            <span className="text-xs tracking-wider uppercase font-semibold">Kayda Başlamak İçin Butona Tıkla</span>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-1 w-full h-full">
            {levels.map((level, index) => {
              const totalBars = levels.length;
              const playIndex = Math.floor(playProgress * (totalBars - 1));
              const isCurrent = index === playIndex && isPlaying;
              const isPassed = index < playIndex;

              return (
                <div
                  key={index}
                  className={`flex-1 rounded-full transition-all duration-75 ease-out
                  ${isCurrent
                      ? isDark
                        ? 'bg-white shadow-[0_0_15px_rgba(255,255,255,0.8)] scale-110'
                        : 'bg-purple-700 shadow-[0_0_15px_rgba(147,51,234,0.8)] scale-110'
                      : isPassed
                        ? isDark
                          ? 'bg-indigo-500/40'
                          : 'bg-purple-400'
                        : isDark
                          ? 'bg-gradient-to-t from-indigo-600 to-purple-500'
                          : 'bg-gradient-to-t from-purple-500 to-indigo-500'
                    }`}
                  style={{
                    height: `${Math.max(4, level * 100)}px`,
                    opacity: isCurrent ? 1 : isPassed ? 0.6 : 0.4 + level * 0.6,
                    transform: isCurrent ? 'scaleY(1.1)' : 'scaleY(1)',
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      <div className={`font-mono text-4xl font-bold mb-6 tracking-widest drop-shadow-lg ${isDark ? "text-white" : "text-gray-800"
        }`}>
        {recordedUrl && duration > 0
          ? `${formatTime(currentTime || 0)} / ${formatTime(duration)}`
          : formatTime(recordingTime)
        }
      </div>

      <div className="flex items-center gap-6 md:gap-6 relative z-10 mb-6">

        {mode === "record" && !isRecording && !recordedUrl && (
          <button onClick={onStartRecording} className="group relative w-20 h-20 rounded-full flex items-center justify-center bg-white hover:scale-105 transition-all duration-300 shadow-[0_0_40px_rgba(255,255,255,0.3)]">
            <div className="absolute inset-0 rounded-full border border-slate-200 animate-ping opacity-20"></div>
            <FaMicrophone className="text-3xl text-indigo-600 group-hover:text-indigo-800 transition-colors" />
          </button>
        )}

        {isRecording && (
          <button onClick={onStopRecording} className="w-20 h-20 rounded-full flex items-center justify-center bg-red-500 hover:bg-red-600 hover:scale-105 transition-all duration-300 shadow-[0_0_30px_rgba(239,68,68,0.4)]">
            <div className="w-8 h-8 bg-white rounded-md"></div>
          </button>
        )}

        {!isRecording && recordedUrl && (
          <>
            <button onClick={onTogglePlay} className="w-20 h-20 rounded-full bg-indigo-500 hover:bg-indigo-400 flex items-center justify-center text-white shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:scale-105 transition-all">
              {isPlaying ? <FaPause className="text-2xl" /> : <FaPlay className="text-2xl ml-1" />}
            </button>

            <div className="relative" ref={speedMenuRef}>
              <button
                onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}
                className={`w-12 h-12 rounded-full flex items-center justify-center transition-all cursor-pointer ${isDark
                  ? "bg-indigo-600/80 hover:bg-indigo-500 border border-indigo-400/30"
                  : "bg-indigo-500 hover:bg-indigo-400 border border-indigo-300"
                  }`}
              >
                {playbackRate === 1.0 ? (
                  <MdSpeed className={`text-lg ${isDark ? "text-white" : "text-white"}`} />
                ) : (
                  <span className="text-xs font-bold text-white">{playbackRate}x</span>
                )}
              </button>

              {isSpeedMenuOpen && (
                <div className={`absolute bottom-full mb-3 left-1/2 -translate-x-1/2 w-32 rounded-xl shadow-2xl overflow-hidden py-1 z-[100] ${isDark ? "bg-slate-800 border border-slate-700" : "bg-white border border-gray-200"
                  }`}>
                  {speedOptions.map((rate) => (
                    <button
                      key={rate}
                      onClick={() => { onSpeedChange(rate); setIsSpeedMenuOpen(false); }}
                      className={`block w-full text-center px-4 py-2.5 text-sm font-medium transition-colors duration-150
                        ${rate === playbackRate
                          ? 'bg-indigo-600 text-white'
                          : isDark
                            ? 'text-white hover:bg-slate-700'
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                    >
                      {rate}x {t('speed') || 'Hız'}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Analyze Button - Kontrol butonlarının altında */}
      {!isRecording && recordedUrl && (
        <button
          onClick={onAnalyze}
          className={`mt-4 px-8 py-4 rounded-xl text-white font-bold text-lg shadow-lg hover:scale-105 transition-all duration-300 flex items-center gap-3 ${isDark
            ? "bg-gradient-to-r from-indigo-600 to-purple-600 hover:shadow-indigo-500/40"
            : "bg-gradient-to-r from-indigo-500 to-purple-500 hover:shadow-indigo-400/40"
            }`}
        >
          <FaMagic className="text-xl" />
          {t('analyze_audio') || 'Ses Analizini Yap'}
        </button>
      )}
    </div>
  );
};

export default AudioPlayer;