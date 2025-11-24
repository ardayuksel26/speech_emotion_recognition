import { FaMicrophone, FaUpload } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import { useTheme } from "../context/ThemeContext";

const ModeSelection = ({ onSelectMode }) => {
  const { t } = useTranslation();
  const { isDark } = useTheme();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 w-full max-w-2xl animate-fade-in-up px-4">
      <button
        onClick={() => onSelectMode("upload")}
        className={`group relative flex flex-col items-center justify-center p-4 md:p-6 rounded-2xl border transition-all duration-300 h-36 md:h-44 ${
          isDark 
            ? "border-white/5 bg-white/5 hover:bg-indigo-600/20 hover:border-indigo-500/50" 
            : "border-indigo-200 bg-indigo-50 hover:bg-indigo-100 hover:border-indigo-400"
        }`}
      >
        <div className={`w-12 h-12 md:w-16 md:h-16 mb-2 md:mb-4 rounded-full flex items-center justify-center transition-colors ${
          isDark 
            ? "bg-indigo-500/10 group-hover:bg-indigo-500/20" 
            : "bg-indigo-200 group-hover:bg-indigo-300"
        }`}>
          <FaUpload className={`text-xl md:text-2xl group-hover:scale-110 transition-all ${
            isDark 
              ? "text-indigo-400 group-hover:text-indigo-300" 
              : "text-indigo-700 group-hover:text-indigo-800"
          }`} />
        </div>
        <h3 className={`text-base md:text-lg font-bold mb-1 ${isDark ? "text-white" : "text-gray-800"}`}>
          {t('upload_audio')}
        </h3>
        <p className={`text-xs text-center px-2 ${isDark ? "text-slate-400" : "text-gray-600"}`}>
          {t('analyze_existing_recording')}
        </p>
      </button>

      <button
        onClick={() => onSelectMode("record")}
        className={`group relative flex flex-col items-center justify-center p-4 md:p-6 rounded-2xl border transition-all duration-300 h-36 md:h-44 ${
          isDark 
            ? "border-white/5 bg-white/5 hover:bg-purple-600/20 hover:border-purple-500/50" 
            : "border-purple-200 bg-purple-50 hover:bg-purple-100 hover:border-purple-400"
        }`}
      >
        <div className={`w-12 h-12 md:w-16 md:h-16 mb-2 md:mb-4 rounded-full flex items-center justify-center transition-colors ${
          isDark 
            ? "bg-purple-500/10 group-hover:bg-purple-500/20" 
            : "bg-purple-200 group-hover:bg-purple-300"
        }`}>
          <FaMicrophone className={`text-xl md:text-2xl group-hover:scale-110 transition-all ${
            isDark 
              ? "text-purple-400 group-hover:text-purple-300" 
              : "text-purple-700 group-hover:text-purple-800"
          }`} />
        </div>
        <h3 className={`text-base md:text-lg font-bold mb-1 ${isDark ? "text-white" : "text-gray-800"}`}>
          {t('record_audio')}
        </h3>
        <p className={`text-xs text-center px-2 ${isDark ? "text-slate-400" : "text-gray-600"}`}>
          {t('record_with_microphone')}
        </p>
      </button>
    </div>
  );
};

export default ModeSelection;
