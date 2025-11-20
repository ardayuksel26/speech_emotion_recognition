import { useRef } from "react";
import { FaUpload } from "react-icons/fa";

const FileUploader = ({ onFileSelect, isDragging, setIsDragging }) => {
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      onFileSelect(file);
    } else {
      alert("Lütfen geçerli bir ses dosyası bırakın.");
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      onFileSelect(file);
    } else {
      alert("Lütfen geçerli bir ses dosyası seçin.");
    }
  };

  return (
    <div className="w-full max-w-lg animate-fade-in-up">
      <div
        className={`relative group border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer
          ${isDragging 
            ? "border-indigo-500 bg-indigo-500/10 scale-[1.02]" 
            : "border-slate-600 hover:border-indigo-400 hover:bg-slate-800/50"}`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onClick={() => fileInputRef.current.click()}
      >
        <FaUpload className="text-5xl text-indigo-400 mx-auto mb-6 group-hover:scale-110 transition-transform duration-300" />
        <h3 className="text-xl font-semibold text-slate-200 mb-2">Dosyayı Sürükle veya Tıkla</h3>
        <p className="text-slate-400 text-sm">MP3, WAV, OGG dosyaları desteklenir</p>
        
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