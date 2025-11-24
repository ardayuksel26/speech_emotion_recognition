import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { FaUpload } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";

const FileUploader = ({ onFileSelect, isDragging, setIsDragging }) => {
  const { t } = useTranslation();
  const { isDark } = useTheme();
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      onFileSelect(file);
    } else {
      alert(t('drop_valid_audio_file'));
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      onFileSelect(file);
    } else {
      alert(t('select_valid_audio_file'));
    }
  };

  return (
    <div className="w-full max-w-lg animate-fade-in-up">
      <div
        className={`relative group border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer
          ${isDragging 
            ? "border-indigo-500 bg-indigo-500/10 scale-[1.02]" 
            : isDark 
              ? "border-slate-600 hover:border-indigo-400 hover:bg-slate-800/50"
              : "border-gray-300 hover:border-indigo-400 hover:bg-indigo-50"}`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onClick={() => fileInputRef.current.click()}
      >
        <FaUpload className={`text-5xl mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 ${
          isDark ? "text-indigo-400" : "text-indigo-600"
        }`} />
        <h3 className={`text-xl font-semibold mb-2 ${
          isDark ? "text-slate-200" : "text-gray-800"
        }`}>
          {t('drag_or_click_to_upload')}
        </h3>
        <p className={`text-sm ${
          isDark ? "text-slate-400" : "text-gray-600"
        }`}>
          {t('supported_formats')}
        </p>
        
        <input 
          ref={fileInputRef}
          type="file" 
          accept="audio/*" 
          className="hidden" 
          onChange={handleFileChange} 
        />
      </div>
    </div>
  );
};

export default FileUploader;