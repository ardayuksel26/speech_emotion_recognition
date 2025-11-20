import { FaMicrophone, FaUpload } from "react-icons/fa";

const ModeSelection = ({ onSelectMode }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-2xl animate-fade-in-up">
      <button
        onClick={() => onSelectMode("upload")}
        className="group relative flex flex-col items-center justify-center p-10 rounded-2xl border border-white/5 bg-white/5 hover:bg-indigo-600/20 hover:border-indigo-500/50 transition-all duration-300 h-64"
      >
        <div className="w-20 h-20 mb-6 rounded-full bg-indigo-500/10 group-hover:bg-indigo-500/20 flex items-center justify-center transition-colors">
          <FaUpload className="text-3xl text-indigo-400 group-hover:text-indigo-300 group-hover:scale-110 transition-all" />
        </div>
        <h3 className="text-xl font-bold text-white mb-2">Dosya Yükle</h3>
        <p className="text-sm text-slate-400 text-center">Hazır bir ses kaydını analiz et</p>
      </button>

      <button
        onClick={() => onSelectMode("record")}
        className="group relative flex flex-col items-center justify-center p-10 rounded-2xl border border-white/5 bg-white/5 hover:bg-purple-600/20 hover:border-purple-500/50 transition-all duration-300 h-64"
      >
        <div className="w-20 h-20 mb-6 rounded-full bg-purple-500/10 group-hover:bg-purple-500/20 flex items-center justify-center transition-colors">
          <FaMicrophone className="text-3xl text-purple-400 group-hover:text-purple-300 group-hover:scale-110 transition-all" />
        </div>
        <h3 className="text-xl font-bold text-white mb-2">Ses Kaydet</h3>
        <p className="text-sm text-slate-400 text-center">Mikrofon ile şimdi kaydet</p>
      </button>
    </div>
  );
};

export default ModeSelection;