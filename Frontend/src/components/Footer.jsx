import { FaGithub } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";

const Footer = () => {
  const { isDark } = useTheme();

  return (
    // DEĞİŞİKLİKLER:
    // 1. border-t: Üst kısma çizgi çeker.
    // 2. shadow-[...]: Footer'dan yukarıya doğru hafif bir gölge atar (Header etkisi).
    <footer className={`relative z-20 w-full transition-colors duration-300 border-t backdrop-blur-md ${isDark
        ? "bg-slate-900/80 text-white border-slate-700/50 shadow-[0_-5px_15px_rgba(0,0,0,0.3)]"
        : "bg-white/80 text-gray-800 border-indigo-200 shadow-[0_-5px_15px_rgba(0,0,0,0.05)]"
      }`}>
      <div className="w-full py-8">
        <div className="flex flex-col items-center justify-center space-y-6">

          {/* GitHub Logosu */}
          <div className="flex items-center">
            <a
              href="https://github.com/ilhanuzunoglu/Speech_Emotion_Recognition_Project"
              target="_blank"
              rel="noopener noreferrer"
              className={`p-3 rounded-lg transition-all duration-200 ${isDark ? "hover:bg-slate-800" : "hover:bg-indigo-200/50"
                }`}
            >
              <FaGithub className={`text-3xl transition-colors ${isDark
                  ? "text-gray-300 hover:text-white"
                  : "text-gray-700 hover:text-gray-900"
                }`} />
            </a>
          </div>

          {/* Copyright */}
          <div className={`text-sm font-medium ${isDark ? "text-gray-400" : "text-gray-600"}`}>
            © 2024 Sesten Duygu Analizi. Tüm hakları saklıdır.
          </div>

        </div>
      </div>
    </footer>
  );
};

export default Footer;