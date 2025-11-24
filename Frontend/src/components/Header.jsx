import { useState } from "react";
import { useTranslation } from "react-i18next";
import { FaMoon, FaSun, FaChevronDown } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";

const Header = () => {
  const { t, i18n } = useTranslation();
  const { isDark, toggleTheme } = useTheme();
  const [isLangOpen, setIsLangOpen] = useState(false);

  const languages = {
    en: { label: "English", flagUrl: "https://flagcdn.com/gb.svg" },
    tr: { label: "Türkçe", flagUrl: "https://flagcdn.com/tr.svg" },
  };

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setIsLangOpen(false);
  };

  const currentLanguage = i18n.language.split('-')[0];

  // YENİ TASARIM: "SOFT PILL" (Yumuşak Hap) Tarzı
  // - rounded-full: Tam yuvarlak köşeler (Kutuluğu alır)
  // - bg-white/40: Arkaplanı silik beyaz yapar (Sert beyaz değil)
  // - shadow-sm: Çok hafif gölge ekler (Havada duruyor hissi)
  // - backdrop-blur: Arkası hafif buzlu cam gibi olur
  const navButtonStyle = `px-5 py-2 rounded-full text-sm font-semibold transition-all duration-300 shadow-sm hover:shadow-md hover:-translate-y-0.5
    ${isDark 
      ? "bg-slate-800/40 text-gray-200 hover:bg-slate-700 hover:text-white border border-white/10" 
      : "bg-white/50 text-slate-700 hover:bg-white hover:text-indigo-600 border border-white/40"
    }`;

  return (
    <header className={`relative w-full shadow-xl sticky top-0 z-50 transition-all duration-300 flex items-center h-14 md:h-16 ${
      isDark 
        ? "bg-slate-900 text-white border-slate-700" 
        : "bg-gradient-to-r from-indigo-100 to-purple-100 text-gray-800 border-indigo-200"
    }`}>
      
      {/* 1. BAŞLIK (KESİN ORTALAMA) */}
      <h1 className={`absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-lg md:text-xl font-extrabold tracking-tight whitespace-nowrap drop-shadow-md z-10 ${
        isDark ? "text-white" : "text-gray-800"
      }`}>
        {t('title')}
      </h1>

      {/* 2. İÇERİK KONTEYNERİ */}
      <div className="w-full flex items-center justify-between px-4">
        
        {/* SOL TARAFTAKİ MENÜ LİNKLERİ */}
        <nav className="hidden md:flex items-center gap-3 ml-4 md:ml-16 z-20">
          <button className={navButtonStyle}>
            {t('about_us', 'Biz Kimiz')}
          </button>
          
          <button className={navButtonStyle}>
            {t('use_cases', 'Kullanım Senaryoları')}
          </button>
        </nav>
        
        {/* SAĞ BUTON GRUBU */}
        <div className="flex items-center gap-3 md:gap-4 mr-4 md:mr-16 z-20">
          
          {/* 🌍 Dil Seçici */}
          <div className="relative">
            <button
              onClick={() => setIsLangOpen(!isLangOpen)}
              className={`flex items-center gap-2 rounded-full px-3 py-1.5 shadow-md hover:shadow-lg transition-all duration-200 ${
                isDark 
                  ? "bg-slate-700 text-white hover:bg-slate-600" 
                  : "bg-white text-indigo-700 hover:bg-indigo-50"
              }`}
            >
              <img
                src={languages[currentLanguage]?.flagUrl}
                alt={languages[currentLanguage]?.label}
                className="w-5 h-5 rounded-full object-cover"
              />
              <span className="hidden md:inline text-sm font-bold">
                {currentLanguage.toUpperCase()}
              </span>
              <FaChevronDown
                className={`text-xs transition-transform duration-200 ${
                  isLangOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {isLangOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setIsLangOpen(false)}></div>
                <div className="absolute right-0 mt-2 w-40 bg-white text-slate-800 rounded-2xl shadow-xl py-2 z-50">
                  {Object.keys(languages).map((code) => (
                    <button
                      key={code}
                      onClick={() => changeLanguage(code)}
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-blue-50 text-sm transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <img
                          src={languages[code].flagUrl}
                          alt={languages[code].label}
                          className="w-6 h-6 rounded-full object-cover"
                        />
                        <span className="font-medium">{languages[code].label}</span>
                      </div>
                      {currentLanguage === code && (
                        <span className="text-green-500 font-bold">✓</span>
                      )}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* 🌙 / ☀️ Tema Switch */}
          <button
            onClick={toggleTheme}
            className={`relative w-12 h-6 rounded-full transition-colors duration-300 focus:outline-none border border-white/30 shadow-inner ${
              isDark ? "bg-slate-800" : "bg-indigo-100"
            }`}
          >
            <div
                className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full shadow-md transform transition-transform duration-300 flex items-center justify-center text-[10px]
                ${isDark ? "translate-x-6 bg-slate-700 text-indigo-300" : "translate-x-0 bg-white text-amber-500"}`}
            >
                {isDark ? <FaMoon /> : <FaSun />}
            </div>
          </button>

        </div>
      </div>
    </header>
  );
};

export default Header;