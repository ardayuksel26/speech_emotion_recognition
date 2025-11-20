import { useState } from "react";
import { FaMoon, FaSun, FaChevronDown } from "react-icons/fa";

const Header = () => {
  const [isDark, setIsDark] = useState(false);
  const [selectedLang, setSelectedLang] = useState("EN");
  const [isLangOpen, setIsLangOpen] = useState(false);

  const languages = {
    EN: { label: "English", flagUrl: "https://flagcdn.com/gb.svg" },
    TR: { label: "Türkçe", flagUrl: "https://flagcdn.com/tr.svg" },
  };

  return (
    <header className="w-full h-16 bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-xl border-b-4 border-blue-800 sticky top-0 z-50">
      <div className="w-full h-full px-4 md:px-6 flex items-center justify-between">

        {/* SOL KISIM (Boşluk) */}
        <div className="w-1/4"></div>

        {/* ORTA BAŞLIK */}
        <h1 className="text-xl md:text-2xl font-bold text-center whitespace-nowrap drop-shadow-md">
          Sesten Duygu Analizi
        </h1>

        {/* SAĞ KISIM */}
        <div className="w-1/4 flex items-center justify-end gap-3 md:gap-4">

          {/* 🌍 Dil Seçici (Eski Tasarım) */}
          <div className="relative">
            <button
              onClick={() => setIsLangOpen(!isLangOpen)}
              className="flex items-center gap-2 bg-white text-blue-700 rounded-full px-3 py-2 shadow-md hover:shadow-lg hover:bg-blue-50 transition-all duration-200"
            >
              <img
                src={languages[selectedLang].flagUrl}
                alt={languages[selectedLang].label}
                className="w-5 h-5 md:w-6 md:h-6 rounded-full object-cover"
              />
              <span className="hidden md:inline text-sm md:text-base font-bold">
                {languages[selectedLang].label.toUpperCase()}
              </span>
              <FaChevronDown
                className={`text-xs md:text-sm transition-transform duration-200 ${
                  isLangOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {isLangOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setIsLangOpen(false)}></div>
                <div className="absolute right-0 mt-2 w-40 bg-white text-slate-800 rounded-2xl shadow-xl py-2 z-50">
                  {["EN", "TR"].map((code) => (
                    <button
                      key={code}
                      onClick={() => {
                        setSelectedLang(code);
                        setIsLangOpen(false);
                      }}
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
                      {selectedLang === code && (
                        <span className="text-green-500 font-bold">✓</span>
                      )}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* 🌙 / ☀️ Tema Switch (Yeni Modern Tasarım - Korundu) */}
          <button
            onClick={() => setIsDark(!isDark)}
            className={`relative w-14 h-7 rounded-full transition-colors duration-300 focus:outline-none border border-white/30 shadow-inner ${
              isDark ? "bg-slate-800" : "bg-indigo-100"
            }`}
          >
            <div
                className={`absolute top-0.5 left-0.5 w-6 h-6 rounded-full shadow-md transform transition-transform duration-300 flex items-center justify-center text-xs
                ${isDark ? "translate-x-7 bg-slate-700 text-indigo-300" : "translate-x-0 bg-white text-amber-500"}`}
            >
                {isDark ? <FaMoon size={10} /> : <FaSun size={12} />}
            </div>
          </button>

        </div>
      </div>
    </header>
  );
};

export default Header;