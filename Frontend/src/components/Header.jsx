import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { FaMoon, FaSun, FaChevronDown, FaBars, FaTimes, FaInfoCircle, FaBullseye, FaFlask, FaCogs } from "react-icons/fa";
import { useTheme } from "../context/ThemeContext";
import { Link, useLocation } from "react-router-dom";
import { clsx } from "clsx";
import { createPortal } from "react-dom";

const Header = () => {
  const { t, i18n } = useTranslation();
  const { isDark, toggleTheme } = useTheme();
  const [isLangOpen, setIsLangOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (isMenuOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => { document.body.style.overflow = "unset"; };
  }, [isMenuOpen]);

  const languages = {
    en: { label: "English", flagUrl: "https://flagcdn.com/gb.svg" },
    tr: { label: "Türkçe", flagUrl: "https://flagcdn.com/tr.svg" },
  };

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setIsLangOpen(false);
  };

  const currentLanguage = i18n.language.split('-')[0];

  const navLinks = [
    { key: 'experimental_models', path: '/experimental', icon: <FaFlask /> },
    { key: 'about_us', path: '/about', icon: <FaInfoCircle /> },
    { key: 'use_cases', path: '/use-cases', icon: <FaBullseye /> },
    { key: 'technical_info', path: '/technical-info', icon: <FaCogs /> },
  ];

  return (
    <header className={clsx(
      "w-full shadow-[0_4px_30px_rgba(0,0,0,0.1)] fixed top-0 left-0 right-0 z-30 transition-all duration-300 flex items-center h-16 font-sans px-4 md:px-8 justify-between backdrop-blur-md",
      isDark ? "bg-slate-900/80 text-white border-b border-slate-700/50" : "bg-white/80 text-gray-800 border-b border-indigo-200"
    )}>
      {/* Mobile Spacer to ensure justify-between pushes controls to the right */}
      <div className="w-8 h-8 lg:hidden"></div>

      {/* LEFT: Logo / Title */}
      <Link to="/" className="logo-desktop-offset z-20 flex items-center gap-2 group no-underline absolute left-1/2 -translate-x-1/2 lg:relative lg:left-0 lg:translate-x-0">
        <h1 className={clsx(
          "text-lg md:text-xl lg:text-2xl font-extrabold tracking-tight whitespace-nowrap drop-shadow-sm transition-transform",
          isDark ? "text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-100 to-indigo-200" : "text-gray-800"
        )}>
          {t('title')}
        </h1>
      </Link>

      {/* CENTER: Desktop Navigation */}
      <nav className="hidden lg:flex items-center gap-6 absolute left-1/2 transform -translate-x-1/2">
        {navLinks.map((link) => (
          <Link
            key={link.key}
            to={link.path}
            className={clsx(
              "text-sm font-semibold transition-colors duration-200 flex items-center gap-2 no-underline hover:no-underline",
              location.pathname === link.path
                ? (isDark ? "text-indigo-300" : "text-indigo-600")
                : (isDark ? "text-slate-300 hover:text-white" : "text-gray-600 hover:text-indigo-600")
            )}
          >
            {t(link.key)}
          </Link>
        ))}
      </nav>

      {/* RIGHT: Controls (Lang, Theme, Hamburger) */}
      <div className="flex items-center gap-3 z-20 ml-auto lg:ml-0">

        {/* Language Switcher (Desktop Only) */}
        <div className="relative hidden lg:block">
            <button
                onClick={() => setIsLangOpen(!isLangOpen)}
                className={clsx(
                    "flex items-center justify-center gap-2.5 rounded-full min-h-[40px] px-6 shadow-sm hover:shadow-md transition-all duration-200 border overflow-hidden",
                    isDark ? "bg-slate-800 text-white hover:bg-slate-700 border-slate-700" : "bg-white text-indigo-700 hover:bg-indigo-50 border-indigo-100"
                )}
                style={{ paddingLeft: '20px', paddingRight: '20px' }}
            >
            <img
              src={languages[currentLanguage]?.flagUrl}
              alt={languages[currentLanguage]?.label}
              className="w-6 h-4 rounded-[2px] object-cover shadow-sm bg-slate-200"
            />
            <span className="hidden md:block text-base font-bold">
              {currentLanguage.toUpperCase()}
            </span>
            <FaChevronDown className={clsx("text-sm transition-transform duration-200", isLangOpen && "rotate-180")} />
          </button>

          {isLangOpen && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setIsLangOpen(false)}></div>
              <div
                className="absolute right-0 mt-2 z-50 overflow-hidden"
                style={{
                  width: '190px',
                  backgroundColor: isDark ? '#1e293b' : '#ffffff',
                  color: isDark ? '#f1f5f9' : '#1e293b',
                  border: isDark ? '1px solid #334155' : '1px solid #e2e8f0',
                  borderRadius: '8px',
                  boxShadow: isDark ? '0 10px 40px rgba(0,0,0,0.5)' : '0 10px 40px rgba(0,0,0,0.12)',
                }}
              >
                {Object.keys(languages).map((code, index) => (
                  <button
                    key={code}
                    onClick={() => changeLanguage(code)}
                    className="w-full flex items-center justify-between text-sm"
                    style={{
                      color: 'inherit',
                      backgroundColor: 'transparent',
                      border: 'none',
                      padding: '12px 16px',
                      borderBottom: index < Object.keys(languages).length - 1
                        ? (isDark ? '1px solid #334155' : '1px solid #f1f5f9')
                        : 'none',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = isDark ? '#334155' : '#f1f5f9'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                  >
                    <div className="flex items-center gap-3">
                      <img src={languages[code].flagUrl} alt="" style={{ width: '22px', height: '22px', borderRadius: '4px', objectFit: 'cover' }} />
                      <span style={{ fontWeight: 500 }}>{languages[code].label}</span>
                    </div>
                    {currentLanguage === code && <span style={{ color: '#22c55e', fontWeight: 'bold' }}>✓</span>}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Theme Switcher */}
        <button
          onClick={toggleTheme}
          className="hidden lg:block focus:outline-none"
          style={{
            position: 'relative',
            width: '76px',
            height: '40px',
            borderRadius: '9999px',
            backgroundColor: isDark ? '#1e293b' : '#e0e7ff',
            border: '1px solid rgba(255,255,255,0.2)',
            boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)',
            transition: 'background-color 0.3s',
            cursor: 'pointer'
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: '4px',
              left: '4px',
              width: '32px',
              height: '32px',
              borderRadius: '9999px',
              backgroundColor: isDark ? '#334155' : '#ffffff',
              boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
              transform: isDark ? 'translateX(36px)' : 'translateX(0)',
              transition: 'transform 0.3s',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '15px',
              color: isDark ? '#818cf8' : '#f59e0b',
            }}
          >
            {isDark ? <FaMoon /> : <FaSun />}
          </div>
        </button>

        {/* Mobile Hamburger (Hidden on Desktop) */}
        <button
          onClick={() => setIsMenuOpen(true)}
          className={clsx(
            "lg:hidden p-2.5 rounded-lg transition-all duration-200 bg-transparent flex items-center justify-center border",
            isDark ? "text-slate-200 border-slate-700 hover:bg-slate-800" : "text-slate-700 border-slate-200 hover:bg-slate-100"
          )}
        >
          <FaBars className="text-xl" />
        </button>
      </div>

      {/* Mobile Menu Drawer via Portal to escape backdrop-filter containing block */}
      {createPortal(
        <div className={clsx(
          "fixed inset-0 z-[9999] transform transition-transform duration-300 lg:hidden",
          isMenuOpen ? "translate-x-0" : "translate-x-full"
        )}>
          {/* Backdrop */}
          <div
            className={clsx("absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity", isMenuOpen ? "opacity-100" : "opacity-0 pointer-events-none")}
            onClick={() => setIsMenuOpen(false)}
          />

          {/* Drawer Content */}
          <div className={clsx(
            "absolute right-0 h-full w-full shadow-2xl flex flex-col p-6 md:p-8 transition-colors duration-300 overflow-y-auto",
            isDark ? "bg-slate-900 text-white" : "bg-white text-slate-900"
          )}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '2rem', marginTop: '1.75rem', paddingRight: '1.75rem', flexShrink: 0 }}>
              <button
                onClick={() => setIsMenuOpen(false)}
                className="p-2 transition-colors bg-transparent border-none outline-none"
              >
                <FaTimes size={24} className={isDark ? "text-white" : "text-slate-800"} />
              </button>
            </div>

            <nav className="flex flex-col gap-8 mb-12">
              {navLinks.map((link) => (
                <Link
                  key={link.key}
                  to={link.path}
                  onClick={() => setIsMenuOpen(false)}
                  className={clsx(
                    "flex justify-between items-center text-xl tracking-wide font-medium no-underline transition-colors",
                    location.pathname === link.path
                      ? (isDark ? "text-white" : "text-black")
                      : (isDark ? "text-slate-300 hover:text-white" : "text-slate-700 hover:text-black")
                  )}
                  style={{ paddingLeft: '3.5rem', paddingRight: '2.5rem' }}
                >
                  <span>{t(link.key)}</span>
                  <span style={{ fontSize: '0.75rem', opacity: 0.4 }}>❯</span>
                </Link>
              ))}
            </nav>

            <div className="mt-auto flex flex-col gap-10 pb-10 shrink-0" style={{ paddingTop: '3.5rem', paddingLeft: '3.5rem', paddingRight: '2.5rem' }}>
              {/* Language Switch */}
              <div className="flex flex-col gap-5">
                <span className={clsx("font-bold text-lg", isDark ? "text-slate-300" : "text-slate-700")}>{t('language')}</span>
                <div className="flex gap-4 w-full">
                  {Object.keys(languages).map(code => (
                    <button
                      key={code}
                      onClick={() => { i18n.changeLanguage(code); }}
                      className={clsx(
                        "flex-1 flex items-center justify-center gap-2.5 rounded-full min-h-[44px] shadow-sm transition-all duration-200 border outline-none",
                        currentLanguage === code
                          ? (isDark ? "bg-slate-800 text-white border-slate-700" : "bg-white text-indigo-700 border-indigo-100")
                          : (isDark ? "bg-transparent text-slate-400 border-slate-700 hover:bg-slate-800" : "bg-transparent text-gray-500 border-gray-200 hover:bg-gray-50")
                      )}
                    >
                      <img src={languages[code].flagUrl} alt="" className="w-5 h-3.5 rounded-[2px] object-cover shadow-sm bg-slate-200" />
                      <span className="font-bold uppercase text-sm tracking-wide">{code}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Theme Switch */}
              <div className="flex justify-between items-center w-full">
                <span className={clsx("font-bold text-lg", isDark ? "text-slate-300" : "text-slate-700")}>{t('theme')}</span>
                <button
                  onClick={toggleTheme}
                  className="focus:outline-none"
                  style={{
                    position: 'relative', width: '68px', height: '36px', borderRadius: '9999px',
                    backgroundColor: isDark ? '#1e293b' : '#e0e7ff',
                    border: '1px solid rgba(255,255,255,0.2)',
                    boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.1)', cursor: 'pointer'
                  }}
                >
                  <div style={{
                    position: 'absolute', top: '3px', left: '3px', width: '28px', height: '28px', borderRadius: '9999px',
                    backgroundColor: isDark ? '#334155' : '#ffffff',
                    transform: isDark ? 'translateX(32px)' : 'translateX(0)', transition: 'transform 0.3s',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px', color: isDark ? '#818cf8' : '#f59e0b'
                  }}>
                    {isDark ? <FaMoon /> : <FaSun />}
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>, document.body
      )}

    </header>
  );
};

export default Header;