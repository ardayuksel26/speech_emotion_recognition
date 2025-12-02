import { useState } from "react";
import { useTranslation } from "react-i18next";
import { FaMoon, FaSun, FaChevronDown, FaBars, FaTimes, FaInfoCircle, FaBullseye, FaGlobe } from "react-icons/fa";
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
    <header className={`relative w-full shadow-xl sticky top-0 z-30 transition-all duration-300 flex items-center h-14 md:h-16 font-sans ${isDark
        ? "bg-slate-900 text-white border-slate-700"
        : "bg-gradient-to-r from-indigo-100 to-purple-100 text-gray-800 border-indigo-200"
      }`}>

      {/* BAŞLIK (KESİN ORTALAMA) */}
      <h1 className={`absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-lg md:text-xl font-extrabold tracking-tight whitespace-nowrap drop-shadow-md z-10 ${isDark ? "text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-100 to-indigo-200" : "text-gray-800"
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
              className={`flex items-center gap-2 rounded-full px-3 py-1.5 shadow-md hover:shadow-lg transition-all duration-200 ${isDark
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
                className={`text-xs transition-transform duration-200 ${isLangOpen ? "rotate-180" : ""
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
            className={`relative w-12 h-6 rounded-full transition-colors duration-300 focus:outline-none border border-white/30 shadow-inner ${isDark ? "bg-slate-800" : "bg-indigo-100"
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
            className={`p-3 rounded-lg transition-all duration-200 hover:scale-105 flex items-center justify-center ${isDark
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
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-opacity duration-300"
            onClick={closeMenu}
          />

          {/* Menu Dropdown - Right Side Panel */}
          <aside className={`fixed top-0 right-0 h-full w-80 max-w-[calc(100%-2rem)] transform transition-transform duration-300 ease-in-out z-50 shadow-2xl ${isMenuAnimating ? 'translate-x-0' : 'translate-x-full'
            } ${isDark
              ? "bg-slate-900/95 border-l border-white/10 backdrop-blur-xl"
              : "bg-white/95 border-l border-white/40 backdrop-blur-xl"
            }`}>

            <div className="flex h-full flex-col p-6">
              {/* Header Section */}
              <div className={`flex items-center justify-between pb-6 border-b relative ${isDark ? "border-white/10" : "border-gray-200"}`}>
                <h1 className={`text-xl font-extrabold tracking-tight ${isDark ? "text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-100 to-indigo-200" : "text-gray-800"
                  }`}>
                  {t('title')}
                </h1>
                <button
                  onClick={closeMenu}
                  className={`p-2 rounded-full transition-colors duration-200 ${isDark
                      ? "text-slate-400 hover:text-white hover:bg-white/10"
                      : "text-slate-500 hover:text-slate-800 hover:bg-gray-100"
                    }`}
                >
                  <FaTimes className="text-xl" />
                </button>
              </div>

              {/* Navigation Links */}
              <nav className="flex flex-col gap-3 py-8 flex-1">
                <button
                  onClick={closeMenu}
                  className={`group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 ${isDark
                      ? "bg-white/5 hover:bg-white/10 text-slate-200 hover:text-white border border-white/5 hover:border-white/10"
                      : "bg-gray-50 hover:bg-white text-slate-700 hover:text-indigo-600 border border-gray-100 hover:border-indigo-100 shadow-sm hover:shadow-md"
                    }`}
                >
                  <div className={`p-2 rounded-lg ${isDark ? "bg-indigo-500/20 text-indigo-300" : "bg-indigo-100 text-indigo-600 group-hover:bg-indigo-600 group-hover:text-white transition-colors"}`}>
                    <FaInfoCircle className="text-lg" />
                  </div>
                  <p className="text-base font-semibold">{t('about_us')}</p>
                </button>
                <button
                  onClick={closeMenu}
                  className={`group flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 ${isDark
                      ? "bg-white/5 hover:bg-white/10 text-slate-200 hover:text-white border border-white/5 hover:border-white/10"
                      : "bg-gray-50 hover:bg-white text-slate-700 hover:text-indigo-600 border border-gray-100 hover:border-indigo-100 shadow-sm hover:shadow-md"
                    }`}
                >
                  <div className={`p-2 rounded-lg ${isDark ? "bg-indigo-500/20 text-indigo-300" : "bg-indigo-100 text-indigo-600 group-hover:bg-indigo-600 group-hover:text-white transition-colors"}`}>
                    <FaBullseye className="text-lg" />
                  </div>
                  <p className="text-base font-semibold">{t('use_cases')}</p>
                </button>
              </nav>

              {/* Settings Section */}
              <div className={`flex flex-col gap-4 pt-6 border-t ${isDark ? "border-white/10" : "border-gray-200"}`}>
                <h3 className={`text-sm font-bold uppercase tracking-wider px-2 ${isDark ? "text-slate-400" : "text-slate-500"
                  }`}>
                  {currentLanguage === 'tr' ? 'Ayarlar' : 'Settings'}
                </h3>

                {/* Language Setting */}
                <div className={`flex items-center justify-between p-4 rounded-xl border ${isDark ? "bg-white/5 border-white/5" : "bg-white border-gray-100 shadow-sm"
                  }`}>
                  <div className="flex items-center gap-3">
                    <div className={`flex items-center justify-center rounded-lg shrink-0 size-10 ${isDark ? "bg-slate-800 text-slate-300" : "bg-indigo-50 text-indigo-600"
                      }`}>
                      <FaGlobe className="text-xl" />
                    </div>
                    <p className={`text-base font-medium ${isDark ? "text-slate-200" : "text-slate-700"
                      }`}>
                      {currentLanguage === 'tr' ? 'Dil' : 'Language'}
                    </p>
                  </div>

                  <div className={`flex items-center rounded-lg p-1 ${isDark ? "bg-black/20" : "bg-gray-100"
                    }`}>
                    <button
                      onClick={() => changeLanguage('tr')}
                      className={`px-3 py-1.5 rounded-md text-sm font-bold transition-all ${currentLanguage === 'tr'
                          ? isDark
                            ? "bg-slate-700 text-white shadow-sm"
                            : "bg-white text-indigo-600 shadow-sm"
                          : isDark ? "text-slate-400 hover:text-slate-200" : "text-slate-500 hover:text-slate-700"
                        }`}
                    >
                      TR
                    </button>
                    <button
                      onClick={() => changeLanguage('en')}
                      className={`px-3 py-1.5 rounded-md text-sm font-bold transition-all ${currentLanguage === 'en'
                          ? isDark
                            ? "bg-slate-700 text-white shadow-sm"
                            : "bg-white text-indigo-600 shadow-sm"
                          : isDark ? "text-slate-400 hover:text-slate-200" : "text-slate-500 hover:text-slate-700"
                        }`}
                    >
                      EN
                    </button>
                  </div>
                </div>

                {/* Theme Setting */}
                <div className={`flex items-center justify-between p-4 rounded-xl border ${isDark ? "bg-white/5 border-white/5" : "bg-white border-gray-100 shadow-sm"
                  }`}>
                  <div className="flex items-center gap-3">
                    <div className={`flex items-center justify-center rounded-lg shrink-0 size-10 ${isDark ? "bg-slate-800 text-slate-300" : "bg-indigo-50 text-indigo-600"
                      }`}>
                      {isDark ? <FaMoon className="text-xl" /> : <FaSun className="text-xl" />}
                    </div>
                    <p className={`text-base font-medium ${isDark ? "text-slate-200" : "text-slate-700"
                      }`}>
                      {currentLanguage === 'tr' ? 'Tema' : 'Theme'}
                    </p>
                  </div>

                  <label className={`relative flex h-[28px] w-[48px] cursor-pointer items-center rounded-full border-none p-1 transition-colors duration-300 ${isDark ? "bg-indigo-600" : "bg-slate-300"
                    }`}
                  >
                    <div
                      className={`h-[20px] w-[20px] rounded-full bg-white shadow-sm transition-transform duration-300 ${isDark ? "translate-x-[20px]" : "translate-x-0"
                        }`}
                    />
                    <input
                      type="checkbox"
                      checked={isDark}
                      onChange={toggleTheme}
                      className="invisible absolute"
                    />
                  </label>
                </div>
              </div>
            </div>

          </aside>
        </>
      )}
    </header>
  );
};

export default Header;