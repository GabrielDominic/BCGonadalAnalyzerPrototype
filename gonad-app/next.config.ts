import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },

  async rewrites() {
    const backendUrl = process.env.BACKEND_URL || "https://bcgonadalanalyzerprototype.onrender.com";
    
    return [
      {
        source: "/api/:path*",
        destination: process.env.NODE_ENV === "development" 
        ? "http://127.0.0.1:8000/:path*" : `${backendUrl}/:path*`,
      },
    ];
  } 
};

export default nextConfig;
