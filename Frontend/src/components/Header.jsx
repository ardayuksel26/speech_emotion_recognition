import { useState } from "react";
import { useTranslation } from "react-i18next";
import { FaMoon, FaSun, FaChevronDown, FaBars, FaTimes } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";

const Header = () => {
  const { t, i18n } = useTranslation();
  const { isDark, toggleTheme } = useTheme();
  const [isLangOpen, setIsLangOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isMenuAnimating, setIsMenuAnimating] = useState(false);

  const languages = {
    en: { label: "English", flagUrl: "https://flagcdn.com/gb.svg" },
    tr: { label: "Türkçe", flagUrl: "https://flagcdn.com/tr.svg" },
  };

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setIsLangOpen(false);
  };
  
  const closeMenu = () => {
    setIsMenuAnimating(false);
    setTimeout(() => {
      setIsMenuOpen(false);
    }, 300); // Corresponds to the duration-300 class
  };

  const openMenu = () => {
    setIsMenuOpen(true);
    setTimeout(() => {
      setIsMenuAnimating(true);
    }, 10);
  };

  const currentLanguage = i18n.language.split('-')[0];

  return (
    <header className={`relative w-full shadow-xl sticky top-0 z-30 transition-all duration-300 flex items-center h-14 md:h-16 ${
      isDark 
        ? "bg-slate-900 text-white border-slate-700" 
        : "bg-gradient-to-r from-indigo-100 to-purple-100 text-gray-800 border-indigo-200"
    }`}>
      
      {/* BAŞLIK (KESİN ORTALAMA) */}
      <h1 className={`absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-lg md:text-xl font-extrabold tracking-tight whitespace-nowrap drop-shadow-md z-10 ${
        isDark ? "text-white" : "text-gray-800"
      }`}>
        {t('title')}
      </h1>

      {/* İÇERİK KONTEYNERİ */}
      <div className="w-full flex items-center justify-end px-4 md:px-16">
        
        {/* SAĞ BUTON GRUBU */}
        <div className="flex items-center gap-3 z-20">
          
          {/* Dil Seçici - Sadece Desktop */}
          <div className="hidden md:block relative">
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

          {/* Tema Switch */}
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

          {/* Hamburger Menü Butonu */}
          <button
            onClick={openMenu}
            className={`p-3 rounded-lg transition-all duration-200 hover:scale-105 ${
              isDark 
                ? "bg-slate-800/50 hover:bg-slate-700 text-white border border-slate-600" 
                : "bg-white/90 hover:bg-white text-indigo-700 border border-indigo-300 shadow-sm hover:shadow-md"
            }`}
          >
            <FaBars className="text-lg" />
          </button>

        </div>
      </div>

      {/* Hamburger Menü */}
      {isMenuOpen && (
        <>
          {/* Overlay */}
          <div
            className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm transition-opacity duration-300"
            onClick={closeMenu}
          />

          {/* Menu Dropdown - Right Side Panel */}
          <div className={`fixed top-0 right-0 h-full w-full md:w-80 transform transition-transform duration-300 ease-in-out z-50 backdrop-blur-xl ${
            isMenuAnimating ? 'translate-x-0' : 'translate-x-full'
          } ${
            isDark
              ? "bg-slate-900/95 border-l border-slate-700/50 text-gray-100"
              : "bg-white/95 border-l border-gray-200/50 text-gray-800"
          }`}>
            
            {/* Header Section */}
            <div className={`px-6 py-4 border-b ${isDark ? "border-slate-700/50" : "border-gray-200/50"} flex items-center justify-between`}>
              <h3 className={`text-sm font-bold tracking-wide ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                {currentLanguage === 'tr' ? 'MENÜ' : 'MENU'}
              </h3>
              <button
                onClick={closeMenu}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  isDark 
                    ? "hover:bg-slate-800 text-gray-400 hover:text-white" 
                    : "hover:bg-gray-100 text-gray-500 hover:text-gray-900"
                }`}
              >
                <FaTimes className="text-lg" />
              </button>
            </div>

            {/* Content Container */}
            <div className="px-6 py-6 h-full overflow-y-auto">
              <div className="grid gap-6">
                
                {/* Navigation Links */}
                <div className="space-y-2">
                  <h4 className={`text-xs font-bold tracking-wide mb-3 ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                    {currentLanguage === 'tr' ? 'SAYFALAR' : 'PAGES'}
                  </h4>
                  <a
                    href="#about"
                    onClick={closeMenu}
                    className={`block w-full text-left px-5 py-3 rounded-lg font-medium transition-all duration-200 ${
                      isDark 
                        ? "hover:bg-slate-800/60 text-gray-200 hover:text-white" 
                        : "hover:bg-indigo-50/80 text-gray-700 hover:text-indigo-600"
                    }`}
                  >
                    {t('about_us')}
                  </a>

                  <a
                    href="#use-cases"
                    onClick={closeMenu}
                    className={`block w-full text-left px-5 py-3 rounded-lg font-medium transition-all duration-200 ${
                      isDark 
                        ? "hover:bg-slate-800/60 text-gray-200 hover:text-white" 
                        : "hover:bg-indigo-50/80 text-gray-700 hover:text-indigo-600"
                    }`}
                  >
                    {t('use_cases')}
                  </a>
                </div>

                {/* Settings Section */}
                <div className="space-y-3">
                  <h4 className={`text-xs font-bold tracking-wide mb-3 ${isDark ? "text-gray-400" : "text-gray-500"}`}>
                    {currentLanguage === 'tr' ? 'AYARLAR' : 'SETTINGS'}
                  </h4>

                  {/* Language Toggle */}
                  <div className={`p-1 rounded-lg flex gap-1 ${isDark ? "bg-slate-800/60" : "bg-gray-100/80"}`}>
                    {Object.keys(languages).map((lang) => (
                      <button
                        key={lang}
                        onClick={() => {
                          changeLanguage(lang);
                        }}
                        className={`flex-1 flex items-center justify-center gap-2 py-2 rounded font-semibold transition-all duration-200 ${
                          currentLanguage === lang
                            ? isDark
                              ? "bg-indigo-600 text-white shadow-md"
                              : "bg-white text-indigo-600 shadow-md"
                            : isDark
                              ? "text-gray-400 hover:text-gray-300"
                              : "text-gray-500 hover:text-gray-700"
                        }`}
                      >
                        <img
                          src={languages[lang].flagUrl}
                          alt={languages[lang].label}
                          className="w-4 h-4 rounded-full object-cover"
                        />
                        <span className="text-xs">{languages[lang].label}</span>
                      </button>
                    ))}
                  </div>

                  {/* Theme Toggle */}
                  <button
                    onClick={() => {
                      toggleTheme();
                    }}
                    className={`w-full flex items-center justify-between px-4 py-3 rounded-lg font-semibold text-sm transition-all duration-200 ${
                      isDark 
                        ? "bg-slate-800/60 text-indigo-300 hover:bg-slate-800" 
                        : "bg-indigo-50/80 text-indigo-700 hover:bg-indigo-100"
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      {isDark ? <FaMoon /> : <FaSun />}
                      {isDark ? (currentLanguage === 'tr' ? 'Karanlık Mod' : 'Dark Mode') : (currentLanguage === 'tr' ? 'Aydınlık Mod' : 'Light Mode')}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${isDark ? "bg-indigo-900/40 text-indigo-300" : "bg-indigo-200/60 text-indigo-700"}`}>
                      {currentLanguage === 'tr' ? 'Aktif' : 'Active'}
                    </span>
                  </button>
                </div>

              </div>
            </div>

          </div>
        </>
      )}
    </header>
  );
};

export default Header;