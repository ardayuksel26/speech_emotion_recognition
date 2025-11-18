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
    <header className="w-full h-16 bg-linear-to-r from-blue-600 to-blue-700 text-white shadow-xl border-b-4 border-blue-800 sticky top-0 z-50">
      <div className="w-full h-full px-6 flex items-center justify-center relative">

        {/* ORTA BAŞLIK */}
        <h1 className="text-2xl font-bold text-center">Sesten Duygu Analizi</h1>

        {/* SAĞ KISIM */}
        <div className="absolute right-6 flex items-center">

          {/* 🌍 Dil Seçici */}
          <div className="relative mr-6">
            <button
              onClick={() => setIsLangOpen(!isLangOpen)}
              className="flex items-center gap-3 bg-white text-blue-700 rounded-full px-4 py-2 shadow-md hover:shadow-lg hover:bg-blue-50 transition-all duration-200"
            >
              <img
                src={languages[selectedLang].flagUrl}
                alt={languages[selectedLang].label}
                className="w-6 h-6 rounded-full object-cover"
              />
              <span className="text-base font-bold">
                {languages[selectedLang].label.toUpperCase()}
              </span>
              <FaChevronDown
                className={`text-sm transition-transform duration-200 ${
                  isLangOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {isLangOpen && (
              <div className="absolute right-0 mt-2 w-52 bg-white text-slate-800 rounded-2xl shadow-xl py-2 z-50">
                {["EN", "TR"].map((code) => (
                  <button
                    key={code}
                    onClick={() => {
                      setSelectedLang(code);
                      setIsLangOpen(false);
                    }}
                    className="w-full flex items-center justify-between px-4 py-3 hover:bg-blue-50 text-base"
                  >
                    <div className="flex items-center gap-3">
                      <img
                        src={languages[code].flagUrl}
                        alt={languages[code].label}
                        className="w-6 h-6 rounded-full object-cover"
                      />
                      <span>{languages[code].label}</span>
                    </div>

                    {selectedLang === code && (
                      <span className="text-green-500 text-lg">✓</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* BOŞLUK */}
          <div className="w-10" />

          {/* 🌙 / ☀️ Tema Switch */}
          <div
            onClick={() => setIsDark(!isDark)}
            className="flex items-center cursor-pointer select-none space-x-2"
          >
            <span
              className={`text-xl transition-all duration-300 ${
                !isDark
                  ? "text-yellow-200 scale-110 drop-shadow-[0_0_8px_rgba(250,250,200,0.8)]"
                  : "text-gray-300 scale-90 opacity-70"
              }`}
            >
              🌙
            </span>

            <div
              className={`w-12 h-6 rounded-full flex items-center px-1 transition-all duration-300 ${
                isDark ? "bg-gray-300" : "bg-gray-500"
              }`}
            >
              <div
                className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform duration-300 ${
                  !isDark ? "translate-x-0" : "translate-x-6"
                }`}
              />
            </div>

            <span
              className={`text-xl transition-all duration-300 ${
                isDark
                  ? "text-yellow-300 scale-110 drop-shadow-[0_0_8px_rgba(252,211,77,0.9)]"
                  : "text-gray-300 scale-90 opacity-70"
              }`}
            >
              ☀️
            </span>
          </div>

        </div>
      </div>
    </header>
  );
};

export default Header;
