import { FaGithub, FaInstagram, FaLinkedin } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="w-full bg-gray-800 text-white border-t-4 border-gray-700 py-8">
      <div className="w-full">
        <div className="flex flex-col items-center justify-center space-y-6">
          {/* Sosyal Medya - Ortalı */}
          <div className="flex items-center gap-8">
            <a
              href="https://github.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-lg hover:bg-gray-700 transition-all duration-200"
            >
              <FaGithub className="text-3xl text-gray-300 hover:text-white transition-colors" />
            </a>

            <a
              href="https://instagram.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-lg hover:bg-gray-700 transition-all duration-200"
            >
              <FaInstagram className="text-3xl text-pink-500 hover:text-pink-400 transition-colors" />
            </a>

            <a
              href="https://linkedin.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-lg hover:bg-gray-700 transition-all duration-200"
            >
              <FaLinkedin className="text-3xl text-blue-500 hover:text-blue-400 transition-colors" />
            </a>
          </div>

          {/* Copyright */}
          <div className="text-sm text-gray-300">
            © 2024 Sesten Duygu Analizi. Tüm hakları saklıdır.
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
